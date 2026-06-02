package com.meta.wearable.dat.externalsampleapps.cameraaccess.gemini

import android.util.Base64
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.File
import java.io.FileOutputStream

class LocalAudioService {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val client = OkHttpClient.Builder().build()

    sealed class PipelineEvent {
        data class Transcript(val text: String) : PipelineEvent()
        data class LlmResponse(val text: String) : PipelineEvent()
        data class TtsReady(val pcmData: ByteArray) : PipelineEvent() {
            override fun equals(other: Any?): Boolean =
                other is TtsReady && other.pcmData.contentEquals(pcmData)
            override fun hashCode(): Int = pcmData.contentHashCode()
        }
        data class Error(val message: String) : PipelineEvent()
    }

    private val _events = MutableStateFlow<PipelineEvent?>(null)
    val events: StateFlow<PipelineEvent?> = _events

    /** Called by LiteLLMSessionViewModel to route TTS audio to the TTS manager */
    var onResponseForTts: ((String) -> Unit)? = null

    /** Called when LLM emits a function call */
    var onToolCall: ((ToolCallPayload) -> Unit)? = null

    // Vision frames queue — prepended to the next chat completion message
    private val visionFrames = mutableListOf<String>()

    // ─── Enqueue a vision frame (sent with next chat completion) ────────────────
    fun enqueueVisionFrame(base64Jpeg: String) {
        synchronized(visionFrames) {
            // Keep only the latest frame to avoid huge payloads
            visionFrames.clear()
            visionFrames.add(base64Jpeg)
        }
    }

    // ─── Step 1: PCM16 → OpenWebUI WhisperX STT (multipart upload) ──────────────
    suspend fun transcribe(pcmData: ByteArray): String? {
        return withContext(Dispatchers.IO) {
            try {
                // Write PCM to a temp file (required for multipart upload)
                val tempFile = File.createTempFile("audio", ".wav")
                FileOutputStream(tempFile).use { it.write(pcmData) }

                val requestBody = MultipartBody.Builder()
                    .setType(MultipartBody.FORM)
                    .addFormDataPart("file", tempFile.name, tempFile.asRequestBody("audio/wav".toMediaType()))
                    .addFormDataPart("model", "whisper")
                    .build()

                val request = Request.Builder()
                    .url(LocalAudioConfig.WHISPERX_URL)
                    .addHeader("Authorization", "Bearer ${LocalAudioConfig.OPEN_WEB_UI_API_KEY}")
                    .post(requestBody)
                    .build()

                val response = client.newCall(request).execute()
                val bodyStr = response.body?.string() ?: ""

                tempFile.delete()

                if (!response.isSuccessful) {
                    _events.value = PipelineEvent.Error("OpenWebUI WhisperX error: ${response.code} $bodyStr")
                    return@withContext null
                }

                // OpenWebUI returns { "text": "..." }
                val json = Json { ignoreUnknownKeys = true }
                val body = json.decodeFromString<Map<String, Any>>(bodyStr)
                val text = (body["text"] as? String) ?: ""

                _events.value = PipelineEvent.Transcript(text)
                text
            } catch (e: Exception) {
                _events.value = PipelineEvent.Error("WhisperX error: ${e.message}")
                null
            }
        }
    }

    // ─── Step 2: Transcript → LiteLLM ─────────────────────────────────────────
    suspend fun generateResponse(transcript: String): String? {
        return try {
            val frames = synchronized(visionFrames) { visionFrames.toList() }
            val messages = buildList {
                add(mapOf("role" to "system", "content" to SYSTEM_PROMPT))
                add(mapOf(
                    "role" to "user",
                    "content" to buildList {
                        add(mapOf("type" to "text", "text" to transcript))
                        frames.forEach { frame ->
                            add(mapOf(
                                "type" to "image_url",
                                "image_url" to mapOf("url" to "data:image/jpeg;base64,$frame")
                            ))
                        }
                    }
                ))
            }

            val body = Json.encodeToString(mapOf(
                "model" to "local",
                "messages" to messages,
                "max_tokens" to 512
            ))

            val request = Request.Builder()
                .url("${LocalAudioConfig.LITELLM_BASE_URL}/chat/completions")
                .addHeader("Authorization", "Bearer ${LocalAudioConfig.OPEN_WEB_UI_API_KEY}")
                .addHeader("Content-Type", "application/json")
                .post(body.toRequestBody("application/json".toMediaType()))
                .build()

            val response = client.newCall(request).execute()
            val bodyStr = response.body?.string() ?: "{}"

            val json = Json { ignoreUnknownKeys = true }
            val bodyMap = json.decodeFromString<Map<String, Any>>(bodyStr)

            // Check for tool call in response
            val toolCalls = bodyMap["choices"]?.let { choices ->
                (choices as? List<*>)?.firstOrNull()
                    ?.let { (it as? Map<String, Any>)?.get("message") }
                    ?.let { (it as? Map<String, Any>)?.get("tool_calls") }
            }

            if (toolCalls != null) {
                handleToolCalls(toolCalls)
                return null
            }

            val content = (bodyMap["choices"] as? List<*>)
                ?.firstOrNull()
                ?.let { (it as? Map<String, Any>)?.get("message") }
                ?.let { (it as? Map<String, Any>)?.get("content") } as? String

            _events.value = PipelineEvent.LlmResponse(content ?: "")
            content
        } catch (e: Exception) {
            _events.value = PipelineEvent.Error("LiteLLM error: ${e.message}")
            null
        }
    }

    // ─── Tool call handling ───────────────────────────────────────────────────
    private fun handleToolCalls(toolCalls: Any?) {
        try {
            val calls = (toolCalls as? List<*>)?.mapNotNull { it as? Map<String, Any> } ?: return
            for (call in calls) {
                val id = call["id"] as? String ?: continue
                val func = call["function"] as? Map<String, Any?> ?: continue
                val name = func["name"] as? String ?: continue
                val args = func["arguments"] as? String ?: "{}"
                onToolCall?.invoke(ToolCallPayload(id, name, args))
            }
        } catch (e: Exception) {
            _events.value = PipelineEvent.Error("Tool call parse error: ${e.message}")
        }
    }

    fun sendToolResponse(response: ToolCallResult) {
        scope.launch {
            try {
                val messages = buildList {
                    add(mapOf("role" to "system", "content" to SYSTEM_PROMPT))
                    add(mapOf(
                        "role" to "user",
                        "content" to listOf(mapOf(
                            "type" to "text",
                            "text" to "Continue."
                        ))
                    ))
                }

                val body = Json.encodeToString(mapOf(
                    "model" to "local",
                    "messages" to messages,
                    "max_tokens" to 512
                ))

                val request = Request.Builder()
                    .url("${LocalAudioConfig.LITELLM_BASE_URL}/chat/completions")
                    .addHeader("Authorization", "Bearer ${LocalAudioConfig.OPEN_WEB_UI_API_KEY}")
                    .addHeader("Content-Type", "application/json")
                    .post(body.toRequestBody("application/json".toMediaType()))
                    .build()

                val httpResponse = client.newCall(request).execute()
                val respBodyStr = httpResponse.body?.string() ?: "{}"

                val json = Json { ignoreUnknownKeys = true }
                val respBody = json.decodeFromString<Map<String, Any>>(respBodyStr)
                val content = (respBody["choices"] as? List<*>)
                    ?.firstOrNull()
                    ?.let { (it as? Map<String, Any>)?.get("message") }
                    ?.let { (it as? Map<String, Any>)?.get("content") } as? String

                content?.let {
                    _events.value = PipelineEvent.LlmResponse(it)
                    onResponseForTts?.invoke(it)
                }
            } catch (e: Exception) {
                _events.value = PipelineEvent.Error("Tool response error: ${e.message}")
            }
        }
    }

    // ─── Inject proactive notification text ────────────────────────────────────
    fun injectNotification(text: String) {
        // Synthesize a response from the injected text
        scope.launch {
            _events.value = PipelineEvent.LlmResponse(text)
            onResponseForTts?.invoke(text)
        }
    }

    // ─── Route LLM text → TTS callback ─────────────────────────────────────────
    fun sendToTts(text: String) {
        onResponseForTts?.invoke(text)
    }

    // ─── Full pipeline (audio + optional vision frames) ───────────────────────
    fun runPipeline(pcmData: ByteArray, visionFrames: List<String> = emptyList()) {
        scope.launch {
            // Enqueue vision frames for this pipeline run
            visionFrames.forEach { enqueueVisionFrame(it) }

            val transcript = transcribe(pcmData) ?: return@launch
            val response = generateResponse(transcript) ?: return@launch
            sendToTts(response)
        }
    }

    fun close() {
        client.close()
    }

    companion object {
        private const val SYSTEM_PROMPT = """
            You are a helpful voice assistant wearing smart glasses. Keep responses concise and conversational.
            The user is speaking to you via voice. Respond in a natural, brief way.
            You have access to tools for controlling smart home devices, querying information, and more.
        """.trimIndent()
    }
}

@Serializable
data class ToolCallPayload(
    val id: String,
    val name: String,
    val arguments: String
)

@Serializable
data class ToolCallResult(
    val id: String,
    val result: String
)