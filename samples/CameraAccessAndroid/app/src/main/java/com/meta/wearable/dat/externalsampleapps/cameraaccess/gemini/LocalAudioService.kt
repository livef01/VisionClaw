package com.meta.wearable.dat.externalsampleapps.cameraaccess.gemini

import android.util.Base64
import io.ktor.client.*
import io.ktor.client.engine.cio.*
import io.ktor.client.plugins.contentnegotiation.*
import io.ktor.client.request.*
import io.ktor.client.statement.*
import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import org.json.JSONObject

class LocalAudioService {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val client = HttpClient(CIO) {
        install(ContentNegotiation) { json(Json { ignoreUnknownKeys = true }) }
    }

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

    // ─── Step 1: PCM16 → WhisperX STT ─────────────────────────────────────────
    suspend fun transcribe(pcmData: ByteArray): String? {
        return try {
            val base64Audio = Base64.encodeToString(pcmData, Base64.NO_WRAP)
            val response: HttpResponse = client.post(LocalAudioConfig.WHISPERX_URL) {
                contentType(ContentType.Application.Json)
                setBody(JSONObject(mapOf("audio_base64" to base64Audio)).toString())
            }

            val body = Json.decodeFromString<Map<String, Any>>(response.bodyAsText())
            val text = (body["text"] as? String) ?: ""

            _events.value = PipelineEvent.Transcript(text)
            text
        } catch (e: Exception) {
            _events.value = PipelineEvent.Error("WhisperX error: ${e.message}")
            null
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

            val response: HttpResponse = client.post("${LocalAudioConfig.LITELLM_BASE_URL}/chat/completions") {
                bearerAuth(System.getenv("LITELLM_KEY") ?: "local")
                contentType(ContentType.Application.Json)
                setBody(JSONObject(mapOf(
                    "model" to "local",
                    "messages" to messages,
                    "max_tokens" to 512
                )).toString())
            }

            val body = Json.decodeFromString<Map<String, Any>>(response.bodyAsText())

            // Check for tool call in response
            val toolCalls = body["choices"]?.let { choices ->
                (choices as? List<*>)?.firstOrNull()
                    ?.let { (it as? Map<String, Any>)?.get("message") }
                    ?.let { (it as? Map<String, Any>)?.get("tool_calls") }
            }

            if (toolCalls != null) {
                handleToolCalls(toolCalls)
                return null
            }

            val content = (body["choices"] as? List<*>)
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

                // Continue conversation with tool result
                val httpResponse: HttpResponse = client.post("${LocalAudioConfig.LITELLM_BASE_URL}/chat/completions") {
                    bearerAuth(System.getenv("LITELLM_KEY") ?: "local")
                    contentType(ContentType.Application.Json)
                    setBody(JSONObject(mapOf(
                        "model" to "local",
                        "messages" to messages,
                        "max_tokens" to 512
                    )).toString())
                }

                val respBody = Json.decodeFromString<Map<String, Any>>(httpResponse.bodyAsText())
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