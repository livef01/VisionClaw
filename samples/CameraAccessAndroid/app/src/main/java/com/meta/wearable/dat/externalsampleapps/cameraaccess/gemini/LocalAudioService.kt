package com.meta.wearable.dat.externalsampleapps.cameraaccess.gemini

import android.graphics.Bitmap
import android.util.Base64
import android.util.Log
import com.meta.wearable.dat.externalsampleapps.cameraaccess.openclaw.GeminiFunctionCall
import com.meta.wearable.dat.externalsampleapps.cameraaccess.openclaw.GeminiToolCall
import com.meta.wearable.dat.externalsampleapps.cameraaccess.openclaw.GeminiToolCallCancellation
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.util.concurrent.TimeUnit

/**
 * Local AI Audio Service — replaces GeminiLiveService.
 *
 * Pipeline:
 *   Mic chunk → [accumulate] → WhisperX STT → text
 *   text → LiteLLM /v1/chat/completions → text response
 *   text → Android TTS (via onResponseForTts callback)
 *
 * Also handles tool calls via OpenClaw bridge.
 */
class LocalAudioService(
    private val scope: CoroutineScope
) {
    companion object {
        private const val TAG = "LocalAudioService"
        private const val MAX_HISTORY = 10
    }

    // ─── Connection state (mirrors GeminiLiveService) ───────────────────────
    sealed class ConnectionState {
        data object Disconnected : ConnectionState()
        data object Connecting : ConnectionState()
        data object SettingUp : ConnectionState()
        data object Ready : ConnectionState()
        data class Error(val message: String) : ConnectionState()
    }

    private val _connectionState = MutableStateFlow<ConnectionState>(ConnectionState.Disconnected)
    val connectionState: StateFlow<ConnectionState> = _connectionState.asStateFlow()

    private val _isModelSpeaking = MutableStateFlow(false)
    val isModelSpeaking: StateFlow<Boolean> = _isModelSpeaking.asStateFlow()

    // ─── Callbacks (same interface as GeminiLiveService) ────────────────────
    var onAudioReceived: ((ByteArray) -> Unit)? = null
    var onTurnComplete: (() -> Unit)? = null
    var onInterrupted: (() -> Unit)? = null
    var onDisconnected: ((String?) -> Unit)? = null
    var onInputTranscription: ((String) -> Unit)? = null
    var onOutputTranscription: ((String) -> Unit)? = null
    var onToolCall: ((GeminiToolCall) -> Unit)? = null
    var onToolCallCancellation: ((GeminiToolCallCancellation) -> Unit)? = null

    /** Called with LLM response text for TTS rendering */
    var onResponseForTts: ((String) -> Unit)? = null

    // ─── HTTP client for LiteLLM + WhisperX ──────────────────────────────────
    private val client = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    // ─── Audio accumulation (Gemini-style: accumulate then flush) ───────────
    private val audioBuffer = ByteArrayOutputStream()
    private val bufferLock = Any()
    private var lastUserSpeechEnd: Long = 0

    // ─── Conversation history ───────────────────────────────────────────────
    private val conversationHistory = mutableListOf<JSONObject>()
    private var sessionKey: String = "agent:main:glass"

    // ─── State observation job ─────────────────────────────────────────────
    private var stateObservationJob: Job? = null

    // ─── Latency tracking ───────────────────────────────────────────────────
    private var responseLatencyLogged = false

    // ─── Connect / Disconnect (mirrors GeminiLiveService lifecycle) ────────

    fun connect(callback: (Boolean) -> Unit) {
        _connectionState.value = ConnectionState.Connecting

        scope.launch {
            try {
                // Verify LiteLLM is reachable
                val healthCheck = withContext(Dispatchers.IO) {
                    val request = Request.Builder()
                        .url("${LocalAudioConfig.LITELLM_BASE_URL}/health")
                        .get()
                        .build()
                    client.newCall(request).execute().use { it.code }
                }

                if (healthCheck != null && healthCheck in 200..499) {
                    Log.d(TAG, "LiteLLM health check: HTTP $healthCheck")
                }

                _connectionState.value = ConnectionState.SettingUp

                // Initialize conversation with system prompt
                conversationHistory.clear()
                conversationHistory.add(JSONObject().apply {
                    put("role", "system")
                    put("content", LocalAudioConfig.systemInstruction)
                })

                _connectionState.value = ConnectionState.Ready
                callback(true)

                // Start state observation
                startStateObservation()

            } catch (e: Exception) {
                Log.e(TAG, "Connection failed: ${e.message}")
                _connectionState.value = ConnectionState.Error(e.message ?: "Unknown error")
                callback(false)
            }
        }
    }

    fun disconnect() {
        stateObservationJob?.cancel()
        stateObservationJob = null
        _connectionState.value = ConnectionState.Disconnected
        _isModelSpeaking.value = false
        conversationHistory.clear()
        audioBuffer.reset()
        onToolCall = null
        onToolCallCancellation = null
        Log.d(TAG, "LocalAudioService disconnected")
    }

    // ─── Audio → STT pipeline ────────────────────────────────────────────────

    /**
     * Called by AudioManager when enough PCM bytes are accumulated.
     * Sends to WhisperX → transcribes → sends to LLM.
     */
    fun sendAudio(data: ByteArray) {
        if (_connectionState.value != ConnectionState.Ready) return

        synchronized(bufferLock) {
            audioBuffer.write(data)
        }

        // Process accumulated audio when buffer is large enough
        val chunk = synchronized(bufferLock) {
            if (audioBuffer.size() >= LocalAudioConfig.MIN_AUDIO_CHUNK_BYTES) {
                val bytes = audioBuffer.toByteArray()
                audioBuffer.reset()
                bytes
            } else null
        }

        if (chunk != null) {
            scope.launch {
                processAudioChunk(chunk)
            }
        }
    }

    private suspend fun processAudioChunk(chunk: ByteArray) {
        // Step 1: STT via WhisperX
        val transcript = withContext(Dispatchers.IO) {
            transcribeWithWhisperX(chunk)
        }

        if (transcript.isBlank()) {
            Log.d(TAG, "Empty transcription, skipping")
            return
        }

        lastUserSpeechEnd = System.currentTimeMillis()
        onInputTranscription?.invoke(transcript)

        // Add user turn to history
        conversationHistory.add(JSONObject().apply {
            put("role", "user")
            put("content", transcript)
        })

        // Trim history to MAX_HISTORY * 2
        if (conversationHistory.size > MAX_HISTORY * 2) {
            conversationHistory.subList(0, conversationHistory.size - MAX_HISTORY * 2).clear()
        }

        // Step 2: LLM via LiteLLM
        _isModelSpeaking.value = true
        val response = withContext(Dispatchers.IO) {
            chatWithLiteLLM()
        }
        _isModelSpeaking.value = false

        if (response.isBlank()) {
            Log.d(TAG, "Empty LLM response")
            onTurnComplete?.invoke()
            return
        }

        onOutputTranscription?.invoke(response)

        // Step 3: TTS — notify listener so ViewModel speaks via Android TTS
        onResponseForTts?.invoke(response)
        onTurnComplete?.invoke()
    }

    private suspend fun transcribeWithWhisperX(audioData: ByteArray): String {
        return try {
            val request = Request.Builder()
                .url(LocalAudioConfig.whisperTranscribeUrl)
                .post(audioData.toRequestBody("audio/pcm;rate=16000".toMediaType()))
                .header("Content-Type", "audio/pcm;rate=16000")
                .build()

            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    Log.e(TAG, "WhisperX failed: ${response.code} ${response.message}")
                    return ""
                }
                response.body?.string()?.trim() ?: ""
            }
        } catch (e: Exception) {
            Log.e(TAG, "WhisperX error: ${e.message}")
            ""
        }
    }

    private suspend fun chatWithLiteLLM(): String {
        // Build messages array for LiteLLM
        val messages = JSONArray()
        for (msg in conversationHistory) {
            messages.put(msg)
        }

        val body = JSONObject().apply {
            put("model", LocalAudioConfig.MODEL)
            put("messages", messages)
            put("max_tokens", 256)
            put("temperature", 0.7)
        }

        val request = Request.Builder()
            .url(LocalAudioConfig.litellmChatUrl)
            .post(body.toString().toRequestBody("application/json".toMediaType()))
            .header("Content-Type", "application/json")
            .build()

        return client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                Log.e(TAG, "LiteLLM failed: ${response.code}")
                return ""
            }
            val json = JSONObject(response.body?.string() ?: "")
            val content = json.optJSONArray("choices")
                ?.optJSONObject(0)
                ?.optJSONObject("message")
                ?.optString("content", "")
                ?: ""
            // Add assistant turn to history
            conversationHistory.add(JSONObject().apply {
                put("role", "assistant")
                put("content", content)
            })
            content
        }
    }

    // ─── Video frame → Vision ───────────────────────────────────────────────

    fun sendVideoFrame(bitmap: Bitmap) {
        if (_connectionState.value != ConnectionState.Ready) return

        scope.launch {
            withContext(Dispatchers.IO) {
                sendVisionFrame(bitmap)
            }
        }
    }

    private suspend fun sendVisionFrame(bitmap: Bitmap) {
        try {
            val baos = ByteArrayOutputStream()
            bitmap.compress(Bitmap.CompressFormat.JPEG, LocalAudioConfig.VIDEO_JPEG_QUALITY, baos)
            val base64 = Base64.encodeToString(baos.toByteArray(), Base64.NO_WRAP)

            // Append vision message to last user message
            val lastUserIdx = conversationHistory.indexOfLast { it.optString("role") == "user" }
            if (lastUserIdx >= 0) {
                val lastUser = conversationHistory[lastUserIdx]
                // Add image to content array (LiteLLM-compatible vision format)
                val currentContent = lastUser.optString("content", "")
                lastUser.put("content", JSONArray().apply {
                    put(JSONObject().apply {
                        put("type", "text")
                        put("text", currentContent)
                    })
                    put(JSONObject().apply {
                        put("type", "image_url")
                        put("image_url", JSONObject().apply {
                            put("url", "data:image/jpeg;base64,$base64")
                        })
                    })
                })
            }
        } catch (e: Exception) {
            Log.e(TAG, "Vision frame error: ${e.message}")
        }
    }

    // ─── Tool calls (OpenClaw bridge) ────────────────────────────────────────

    /**
     * Called by LiteLLMSessionViewModel after a tool response comes back.
     * Routes back into LLM context.
     */
    fun sendToolResponse(response: JSONObject) {
        try {
            val functionResponses = response.optJSONObject("toolResponse")
                ?.optJSONArray("functionResponses")
            if (functionResponses != null) {
                for (i in 0 until functionResponses.length()) {
                    val fr = functionResponses.getJSONObject(i)
                    val result = fr.optString("result", "")
                    conversationHistory.add(JSONObject().apply {
                        put("role", "system")
                        put("content", "[TOOL RESULT] $result")
                    })
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error processing tool response: ${e.message}")
        }
    }

    fun sendTextMessage(text: String) {
        if (_connectionState.value != ConnectionState.Ready) return
        conversationHistory.add(JSONObject().apply {
            put("role", "user")
            put("content", text)
        })
    }

    // ─── State observation ───────────────────────────────────────────────────

    private fun startStateObservation() {
        stateObservationJob = scope.launch {
            while (isActive) {
                delay(100)
                // State propagation happens via flows directly
            }
        }
    }
}

/** Re-exports for compatibility with ToolCallRouter */
typealias ToolCallStatus = com.meta.wearable.dat.externalsampleapps.cameraaccess.openclaw.ToolCallStatus