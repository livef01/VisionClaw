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

    /** Callback injected by LiteLLMSessionViewModel to route TTS audio to the TTS manager */
    var onResponseForTts: ((String) -> Unit)? = null

    // --- Step 1: PCM16 → WhisperX STT ---
    suspend fun transcribe(pcmData: ByteArray): String? {
        return try {
            val base64Audio = Base64.encodeToString(pcmData, Base64.NO_WRAP)
            val body = mapOf(
                "audio" to mapOf(
                    "data" to base64Audio,
                    "format" to "s16le",
                    "sample_rate" to 16000,
                    "channels" to 1
                ),
                "model" to "base",
                "language" to "en"
            )

            val response: HttpResponse = client.post(LocalAudioConfig.WHISPERX_URL) {
                contentType(ContentType.Application.Json)
                setBody(Json.encodeToString(mapOf("audio" to base64Audio)))
            }

            val text = Json.decodeFromString<Map<String, String>>(
                response.bodyAsText()
            )["text"] ?: ""

            _events.value = PipelineEvent.Transcript(text)
            text
        } catch (e: Exception) {
            _events.value = PipelineEvent.Error("WhisperX error: ${e.message}")
            null
        }
    }

    // --- Step 2: Transcript → LiteLLM ---
    suspend fun generateResponse(transcript: String, visionFrames: List<String> = emptyList()): String? {
        return try {
            val messages = buildList {
                add(mapOf("role" to "system", "content" to SYSTEM_PROMPT))
                add(mapOf(
                    "role" to "user",
                    "content" to buildList {
                        add(mapOf("type" to "text", "text" to transcript))
                        visionFrames.forEach { frame ->
                            add(mapOf("type" to "image_url", "image_url" to mapOf("url" to "data:image/jpeg;base64,$frame")))
                        }
                    }
                ))
            }

            val response: HttpResponse = client.post("${LocalAudioConfig.LITELLM_BASE_URL}/chat/completions") {
                bearerAuth(System.getenv("LITELLM_KEY") ?: "local")
                contentType(ContentType.Application.Json)
                setBody(Json.encodeToString(mapOf(
                    "model" to "local",
                    "messages" to messages,
                    "max_tokens" to 512
                )))
            }

            val body = Json.decodeFromString<Map<String, Any>>(response.bodyAsText())
            val content = (body["choices"] as? List<*>)?.get(0)
                ?.let { (it as? Map<String, Any>)?.get("message") }
                ?.let { (it as? Map<String, Any>)?.get("content") } as? String

            _events.value = PipelineEvent.LlmResponse(content ?: "")
            content
        } catch (e: Exception) {
            _events.value = PipelineEvent.Error("LiteLLM error: ${e.message}")
            null
        }
    }

    // --- Step 3: Route LLM text → TTS callback ---
    fun sendToTts(text: String) {
        onResponseForTts?.invoke(text)
    }

    // --- Full pipeline ---
    fun runPipeline(pcmData: ByteArray, visionFrames: List<String> = emptyList()) {
        scope.launch {
            val transcript = transcribe(pcmData) ?: return@launch
            val response = generateResponse(transcript, visionFrames) ?: return@launch
            sendToTts(response)
        }
    }

    fun close() {
        client.close()
    }

    companion object {
        private const val SYSTEM_PROMPT = """
            You are a helpful voice assistant. Keep responses concise and conversational.
            The user is speaking to you via voice. Respond in a natural, brief way.
        """.trimIndent()
    }
}