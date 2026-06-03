package com.meta.wearable.dat.externalsampleapps.cameraaccess.gemini

import android.util.Base64
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import org.json.JSONArray
import org.json.JSONObject

/**
 * Pipeline: PCM16 → WhisperX STT → LiteLLM (with tool calls) → TTS callback.
 *
 * Uses OkHttp + org.json (same stack as OpenClawBridge) so the project doesn't
 * pull in an extra HTTP client. Tool-call routing goes through
 * LiteLLMSessionViewModel, which delegates to ToolCallRouter.
 */
class LocalAudioService {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    private val client = OkHttpClient.Builder()
        .readTimeout(120, TimeUnit.SECONDS)
        .connectTimeout(10, TimeUnit.SECONDS)
        .build()

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

    /** Called when LLM emits a function call (invoked once per call) */
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
            val body = JSONObject().put("audio", base64Audio).toString()

            val request = Request.Builder()
                .url(LocalAudioConfig.WHISPERX_URL)
                .post(body.toRequestBody(JSON))
                .build()

            client.newCall(request).execute().use { response ->
                val text = parseTranscript(response)
                _events.value = PipelineEvent.Transcript(text)
                text
            }
        } catch (e: Exception) {
            _events.value = PipelineEvent.Error("WhisperX error: ${e.message}")
            null
        }
    }

    private fun parseTranscript(response: Response): String {
        if (!response.isSuccessful) return ""
        val body = response.body?.string() ?: return ""
        return JSONObject(body).optString("text", "")
    }

    // ─── Step 2: Transcript → LiteLLM ─────────────────────────────────────────
    suspend fun generateResponse(transcript: String): String? {
        return try {
            val frames = synchronized(visionFrames) { visionFrames.toList() }

            val messagesArray = JSONArray()
            messagesArray.put(systemMessage())
            messagesArray.put(userMessageWithFrames(transcript, frames))

            val body = JSONObject().apply {
                put("model", "local")
                put("messages", messagesArray)
                put("max_tokens", 512)
            }.toString()

            val request = Request.Builder()
                .url("${LocalAudioConfig.LITELLM_BASE_URL}/chat/completions")
                .post(body.toRequestBody(JSON))
                .addHeader("Authorization", "Bearer ${System.getenv("LITELLM_KEY") ?: "local"}")
                .build()

            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    _events.value = PipelineEvent.Error("LiteLLM HTTP ${response.code}")
                    return null
                }
                val respBody = response.body?.string().orEmpty()
                val json = JSONObject(respBody)

                // Check for tool call in response
                val toolCalls = firstChoice(json)?.optJSONArray("tool_calls")
                if (toolCalls != null && toolCalls.length() > 0) {
                    handleToolCalls(toolCalls)
                    return null
                }

                val content = firstMessage(json)?.optString("content", "").orEmpty()
                _events.value = PipelineEvent.LlmResponse(content)
                content
            }
        } catch (e: Exception) {
            _events.value = PipelineEvent.Error("LiteLLM error: ${e.message}")
            null
        }
    }

    // ─── Tool call handling ───────────────────────────────────────────────────
    private fun handleToolCalls(toolCalls: JSONArray) {
        try {
            for (i in 0 until toolCalls.length()) {
                val call = toolCalls.optJSONObject(i) ?: continue
                val id = call.optString("id")
                if (id.isEmpty()) continue
                val func = call.optJSONObject("function") ?: continue
                val name = func.optString("name")
                if (name.isEmpty()) continue
                val args = func.optString("arguments", "{}")
                onToolCall?.invoke(ToolCallPayload(id, name, args))
            }
        } catch (e: Exception) {
            _events.value = PipelineEvent.Error("Tool call parse error: ${e.message}")
        }
    }

    /**
     * Continue the conversation with a tool result. The router builds a JSON
     * `toolResponse` envelope and we extract the first function response,
     * inject it as a `role: "tool"` message (OpenAI-compatible), and hand the
     * augmented messages to LiteLLM.
     *
     * Envelope shape (from ToolCallRouter.buildToolResponse):
     *   { "toolResponse": { "functionResponses": [{ "id": "...", "name": "...",
     *                          "response": { "result": "..." } | { "error": "..." } }] } }
     */
    fun sendToolResponse(response: JSONObject) {
        scope.launch {
            try {
                val firstFnResponse = response
                    .optJSONObject("toolResponse")
                    ?.optJSONArray("functionResponses")
                    ?.optJSONObject(0)
                val toolCallId = firstFnResponse?.optString("id", "").orEmpty()
                val resultObj = firstFnResponse?.optJSONObject("response")
                val content = when {
                    resultObj == null -> ""
                    resultObj.has("result") -> resultObj.optString("result")
                    resultObj.has("error") -> "Error: ${resultObj.optString("error")}"
                    else -> resultObj.toString()
                }

                val messagesArray = JSONArray()
                messagesArray.put(systemMessage())
                if (toolCallId.isNotEmpty()) {
                    messagesArray.put(JSONObject().apply {
                        put("role", "tool")
                        put("tool_call_id", toolCallId)
                        put("content", content)
                    })
                } else {
                    // Fall back to a user "continue" prompt when the envelope
                    // shape is unrecognized — preserves prior behavior.
                    messagesArray.put(JSONObject().apply {
                        put("role", "user")
                        put("content", JSONArray().put(JSONObject().apply {
                            put("type", "text")
                            put("text", "Continue.")
                        }))
                    })
                }

                val body = JSONObject().apply {
                    put("model", "local")
                    put("messages", messagesArray)
                    put("max_tokens", 512)
                }.toString()

                val request = Request.Builder()
                    .url("${LocalAudioConfig.LITELLM_BASE_URL}/chat/completions")
                    .post(body.toRequestBody(JSON))
                    .addHeader("Authorization", "Bearer ${System.getenv("LITELLM_KEY") ?: "local"}")
                    .build()

                client.newCall(request).execute().use { httpResponse ->
                    if (!httpResponse.isSuccessful) {
                        _events.value = PipelineEvent.Error("Tool response HTTP ${httpResponse.code}")
                        return@launch
                    }
                    val respBody = httpResponse.body?.string().orEmpty()
                    val json = JSONObject(respBody)
                    val nextContent = firstMessage(json)?.optString("content", "").orEmpty()
                    if (nextContent.isNotEmpty()) {
                        _events.value = PipelineEvent.LlmResponse(nextContent)
                        onResponseForTts?.invoke(nextContent)
                    }
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
        client.dispatcher.executorService.shutdown()
    }

    // ─── JSON helpers ─────────────────────────────────────────────────────────
    private fun systemMessage(): JSONObject = JSONObject().apply {
        put("role", "system")
        put("content", SYSTEM_PROMPT)
    }

    private fun userMessageWithFrames(transcript: String, frames: List<String>): JSONObject {
        val content = JSONArray()
        content.put(JSONObject().apply {
            put("type", "text")
            put("text", transcript)
        })
        for (frame in frames) {
            content.put(JSONObject().apply {
                put("type", "image_url")
                put("image_url", JSONObject().put("url", "data:image/jpeg;base64,$frame"))
            })
        }
        return JSONObject().apply {
            put("role", "user")
            put("content", content)
        }
    }

    private fun firstChoice(json: JSONObject): JSONObject? =
        json.optJSONArray("choices")?.optJSONObject(0)

    private fun firstMessage(json: JSONObject): JSONObject? =
        firstChoice(json)?.optJSONObject("message")

    companion object {
        private val JSON = "application/json; charset=utf-8".toMediaType()
        private val SYSTEM_PROMPT = """
            You are a helpful voice assistant wearing smart glasses. Keep responses concise and conversational.
            The user is speaking to you via voice. Respond in a natural, brief way.
            You have access to tools for controlling smart home devices, querying information, and more.
        """.trimIndent()
    }
}

/** Lightweight payload for one tool call (id + name + JSON-encoded args). */
data class ToolCallPayload(
    val id: String,
    val name: String,
    val arguments: String
)
