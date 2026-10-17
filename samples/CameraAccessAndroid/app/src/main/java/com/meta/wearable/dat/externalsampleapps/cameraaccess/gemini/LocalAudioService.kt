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

    // Conversation history for bounded context
    private val conversationHistory = mutableListOf<Map<String, Any>>()

    companion object {
        /** Maximum number of messages to keep in conversation history (bounded memory) */
        const val MAX_HISTORY_MESSAGES = 20

        /** Pattern to strip reasoning/thinking tags from LLM responses */
        private val THINK_TAG_PATTERN = Regex("<think>.*?</think>", RegexOption.DOT_MATCHES_ALL)

        /** Strips any `<think>...</think>` reasoning blocks from text.
         *  Returns the cleaned text, or empty string if only reasoning tags were present.
         */
        fun stripReasoningTags(text: String): String {
            if (text.isBlank()) return text
            val cleaned = THINK_TAG_PATTERN.replace(text, "").trim()
            return cleaned.takeIf { it.isNotEmpty() } ?: text.trim()
        }

        /** Builds an OpenAI-compatible messages array with system prompt,
         *  bounded conversation history, and optional vision frames.
         *  Keeps only the last N messages to bound memory growth.
         */
        fun buildMessages(
            systemPrompt: String,
            transcript: String,
            frames: List<String>,
            history: List<Map<String, Any>>
        ): JSONArray {
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

            val messagesArray = JSONArray()

            // Always include system message
            val systemMsg = JSONObject()
            systemMsg.put("role", "system")
            systemMsg.put("content", systemPrompt)
            messagesArray.put(systemMsg)

            // Add bounded conversation history (oldest first, then current)
            val recentHistory = if (history.size > MAX_HISTORY_MESSAGES) {
                history.subList(history.size - MAX_HISTORY_MESSAGES + 1, history.size)
            } else {
                history.toList()
            }
            for (msg in recentHistory) {
                messagesArray.put(JSONObject(msg))
            }

            // Add current user message with optional vision frames
            val userMsg = JSONObject()
            userMsg.put("role", "user")
            userMsg.put("content", content)
            messagesArray.put(userMsg)

            return messagesArray
        }

        /**
         * Appends a user+assistant pair to history (one turn).
         * Keeps history bounded by the MAX_HISTORY_MESSAGES limit.
         */
        fun recordTurn(history: MutableList<Map<String, Any>>, userMessage: Map<String, Any>, assistantMessage: Map<String, Any>): MutableList<Map<String, Any>> {
            history.add(userMessage)
            history.add(assistantMessage)

            // Keep bounded
            while (history.size > MAX_HISTORY_MESSAGES * 2) {
                // Remove oldest pair (user + assistant) to maintain pairing
                if (history.size >= 2) {
                    history.removeAt(0)
                    history.removeAt(0)
                } else {
                    break
                }
            }
            return history
        }
    }

    // ─── Enqueue a vision frame (sent with next chat completion) ────────────────
    fun enqueueVisionFrame(base64Jpeg: String) {
        synchronized(visionFrames) {
            // Keep only the latest frame to avoid huge payloads
            visionFrames.clear()
            visionFrames.add(base64Jpeg)
        }
    }

    // ─── Step 1: PCM16 → WhisperX STT (transcribe_base64 endpoint) ─────────────
    suspend fun transcribe(pcmData: ByteArray): String? {
        return try {
            val base64Audio = Base64.encodeToString(pcmData, Base64.NO_WRAP)
            // Use 'audio_base64' key to match the deployed WhisperX contract
            val body = JSONObject().put("audio_base64", base64Audio).toString()

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

            val messagesArray = LocalAudioService.buildMessages(
                systemPrompt = SYSTEM_PROMPT,
                transcript = transcript,
                frames = frames,
                history = conversationHistory.toList()
            )

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

                val rawContent = firstMessage(json)?.optString("content", "").orEmpty()
                // Strip reasoning tags before forwarding to UI/TTS
                val content = stripReasoningTags(rawContent)
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

                // Start with system message
                val systemMsg = JSONObject()
                systemMsg.put("role", "system")
                systemMsg.put("content", SYSTEM_PROMPT)
                messagesArray.put(systemMsg)

                // Add bounded conversation history
                val recentHistory = if (conversationHistory.size > MAX_HISTORY_MESSAGES) {
                    conversationHistory.subList(conversationHistory.size - MAX_HISTORY_MESSAGES + 1, conversationHistory.size)
                } else {
                    conversationHistory.toList()
                }
                for (msg in recentHistory) {
                    messagesArray.put(JSONObject(msg))
                }

                if (toolCallId.isNotEmpty()) {
                    val toolMsg = JSONObject().apply {
                        put("role", "tool")
                        put("tool_call_id", toolCallId)
                        put("content", content)
                    }
                    messagesArray.put(toolMsg)
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
                    val rawContent = firstMessage(json)?.optString("content", "").orEmpty()
                    // Strip reasoning tags before forwarding
                    val content = stripReasoningTags(rawContent)
                    if (content.isNotEmpty()) {
                        _events.value = PipelineEvent.LlmResponse(content)
                        onResponseForTts?.invoke(content)
                    }
                }
            } catch (e: Exception) {
                _events.value = PipelineEvent.Error("Tool response error: ${e.message}")
            }
        }
    }

    // ─── Inject proactive notification text ────────────────────────────────────
    fun injectNotification(text: String) {
        // Strip reasoning tags even from injected notifications
        val cleanedText = stripReasoningTags(text)
        // Synthesize a response from the injected text
        scope.launch {
            _events.value = PipelineEvent.LlmResponse(cleanedText)
            onResponseForTts?.invoke(cleanedText)
        }
    }

    // ─── Route LLM text → TTS callback ─────────────────────────────────────────
    fun sendToTts(text: String) {
        val cleanedText = stripReasoningTags(text)
        onResponseForTts?.invoke(cleanedText)
    }

    /**
     * Records the user input and assistant response in conversation history
     * with bounded memory. Called by the ViewModel after each pipeline round-trip.
     */
    fun recordConversationTurn(userText: String, assistantText: String) {
        val userMsg = mapOf(
            "role" to "user",
            "content" to userText
        )
        val assistantMsg = mapOf(
            "role" to "assistant",
            "content" to stripReasoningTags(assistantText)
        )
        LocalAudioService.recordTurn(conversationHistory, userMsg, assistantMsg)
    }

    /**
     * Exposes the current conversation history for the ViewModel to observe
     * (useful for debugging or persistence).
     */
    fun getConversationHistory(): List<Map<String, Any>> {
        return conversationHistory.toList()
    }

    /**
     * Clears conversation history (useful for session reset).
     */
    fun clearConversationHistory() {
        synchronized(conversationHistory) {
            conversationHistory.clear()
        }
    }

    // ─── Full pipeline (audio + optional vision frames) ───────────────────────
    fun runPipeline(pcmData: ByteArray, visionFrames: List<String> = emptyList()) {
        scope.launch {
            // Enqueue vision frames for this pipeline run
            visionFrames.forEach { enqueueVisionFrame(it) }

            val transcript = transcribe(pcmData) ?: return@launch
            val response = generateResponse(transcript) ?: return@launch

            // Record this turn in conversation history
            recordConversationTurn(transcript, response)

            sendToTts(response)
        }
    }

    fun close() {
        client.dispatcher.executorService.shutdown()
    }

    // ─── JSON helpers ─────────────────────────────────────────────────────────
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
