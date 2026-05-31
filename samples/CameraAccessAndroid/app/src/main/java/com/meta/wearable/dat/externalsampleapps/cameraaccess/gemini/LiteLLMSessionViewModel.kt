package com.meta.wearable.dat.externalsampleapps.cameraaccess.gemini

import android.app.Application
import android.graphics.Bitmap
import android.util.Base64
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.meta.wearable.dat.externalsampleapps.cameraaccess.openclaw.OpenClawBridge
import com.meta.wearable.dat.externalsampleapps.cameraaccess.openclaw.OpenClawEventClient
import com.meta.wearable.dat.externalsampleapps.cameraaccess.openclaw.OpenClawConnectionState
import com.meta.wearable.dat.externalsampleapps.cameraaccess.openclaw.GeminiFunctionCall
import com.meta.wearable.dat.externalsampleapps.cameraaccess.openclaw.ToolCallRouter
import com.meta.wearable.dat.externalsampleapps.cameraaccess.openclaw.ToolCallStatus
import com.meta.wearable.dat.externalsampleapps.cameraaccess.gemini.ToolCallResult
import com.meta.wearable.dat.externalsampleapps.cameraaccess.settings.SettingsManager
import org.json.JSONObject
import com.meta.wearable.dat.externalsampleapps.cameraaccess.stream.StreamingMode
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

// ─── Connection state (mirrors GeminiConnectionState for UI compatibility) ───
sealed class LiteLlmConnectionState {
    data object Disconnected : LiteLlmConnectionState()
    data object Connecting : LiteLlmConnectionState()
    data object Ready : LiteLlmConnectionState()
    data object Error : LiteLlmConnectionState()
}

// ─── UI state (mirrors GeminiUiState — StreamScreen reads these fields) ───────
data class LiteLlmUiState(
    val isGeminiActive: Boolean = false,       // "AI active" toggle in StreamScreen
    val connectionState: LiteLlmConnectionState = LiteLlmConnectionState.Disconnected,
    val isModelSpeaking: Boolean = false,
    val isRecording: Boolean = false,           // continuous listening capture state
    val errorMessage: String? = null,
    val userTranscript: String = "",
    val aiTranscript: String = "",
    val toolCallStatus: ToolCallStatus = ToolCallStatus.Idle,
    val openClawConnectionState: OpenClawConnectionState = OpenClawConnectionState.NotConfigured,
)

class LiteLLMSessionViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "LiteLLMSessionVM"
        private const val VIDEO_FRAME_INTERVAL_MS = 1000L // 1 fps for vision frames
    }

    private val _uiState = MutableStateFlow(LiteLlmUiState())
    val uiState: StateFlow<LiteLlmUiState> = _uiState.asStateFlow()

    private val localAudioManager = LocalAudioManager(application)
    private val localTtsManager = LocalTtsManager(application)
    private val localAudioService = LocalAudioService()
    private val openClawBridge = OpenClawBridge()
    private val eventClient = OpenClawEventClient()

    private var toolCallRouter: ToolCallRouter? = null
    private var lastVideoFrameTime: Long = 0
    private var stateObservationJob: Job? = null
    private var sessionActive = false

    var streamingMode: StreamingMode = StreamingMode.GLASSES

    init {
        localTtsManager.init()

        // Wire TTS callback into the pipeline
        localAudioService.onResponseForTts = { text ->
            localTtsManager.speak(text)
            _uiState.value = _uiState.value.copy(
                aiTranscript = text,
                isModelSpeaking = false,
            )
        }

        // Observe service events → update UI state
        viewModelScope.launch {
            localAudioService.events.collect { event ->
                when (event) {
                    is LocalAudioService.PipelineEvent.Transcript -> {
                        _uiState.value = _uiState.value.copy(userTranscript = event.text)
                    }
                    is LocalAudioService.PipelineEvent.LlmResponse -> {
                        _uiState.value = _uiState.value.copy(aiTranscript = event.text)
                    }
                    is LocalAudioService.PipelineEvent.TtsReady -> {
                        localAudioManager.playPcm16kHz(event.pcmData)
                    }
                    is LocalAudioService.PipelineEvent.Error -> {
                        _uiState.value = _uiState.value.copy(errorMessage = event.message)
                    }
                    null -> {}
                }
            }
        }
    }

    fun startSession() {
        if (sessionActive) return

        if (!LocalAudioConfig.isConfigured()) {
            _uiState.value = _uiState.value.copy(
                errorMessage = "Local audio pipeline not configured. Set LiteLLM and WhisperX endpoints in LocalAudioConfig.kt"
            )
            return
        }

        sessionActive = true
        _uiState.value = _uiState.value.copy(
            isGeminiActive = true,
            connectionState = LiteLlmConnectionState.Connecting,
            errorMessage = null,
        )

        viewModelScope.launch {
            // Check OpenClaw connection
            openClawBridge.checkConnection()
            openClawBridge.resetSession()

            // Wire tool call handling
            toolCallRouter = ToolCallRouter(openClawBridge, viewModelScope)

            localAudioService.onToolCall = { toolCallPayload ->
                val geminiCall = GeminiFunctionCall(
                    id = toolCallPayload.id,
                    name = toolCallPayload.name,
                    args = try {
                        JSONObject(toolCallPayload.arguments).let { obj ->
                            obj.keys().asSequence().associateWith { obj.opt(it) }
                        }
                    } catch (e: Exception) {
                        emptyMap()
                    }
                )
                toolCallRouter?.handleToolCall(geminiCall) { response ->
                    // Parse JSONObject → ToolCallResult
                    val toolResponse = response.optJSONObject("toolResponse")
                        ?.optJSONArray("functionResponses")
                        ?.optJSONObject(0)
                    val resultId = toolResponse?.optString("id") ?: ""
                    val resultObj = toolResponse?.optJSONObject("response")
                    val resultStr = resultObj?.optString("result")
                        ?: resultObj?.optString("error")
                        ?: ""
                    localAudioService.sendToolResponse(ToolCallResult(resultId, resultStr))
                }
            }

            // Start polling state
            stateObservationJob = viewModelScope.launch {
                while (isActive) {
                    delay(100)
                    _uiState.value = _uiState.value.copy(
                        openClawConnectionState = openClawBridge.connectionState.value,
                        toolCallStatus = openClawBridge.lastToolCallStatus.value,
                    )
                }
            }

            // Mark ready (mic capture starts on user press, not on session start)
            _uiState.value = _uiState.value.copy(
                connectionState = LiteLlmConnectionState.Ready,
            )

            // OpenClaw event stream for proactive notifications
            if (SettingsManager.proactiveNotificationsEnabled) {
                eventClient.onNotification = { text ->
                    if (sessionActive) {
                        localAudioService.injectNotification(text)
                    }
                }
                eventClient.connect()
            }
        }
    }

    fun stopSession() {
        sessionActive = false
        eventClient.disconnect()
        toolCallRouter?.cancelAll()
        toolCallRouter = null
        stateObservationJob?.cancel()
        stateObservationJob = null
        _uiState.value = LiteLlmUiState()
    }

    /**
     * Called by StreamViewModel when the user presses the AI toggle.
     * Starts continuous listening: capture → transcribe → respond → TTS.
     */
    fun startListening() {
        if (!sessionActive) return
        if (_uiState.value.isRecording) return

        _uiState.value = _uiState.value.copy(isRecording = true)

        viewModelScope.launch {
            while (sessionActive && _uiState.value.isRecording) {
                val pcmData = localAudioManager.capturePcm16kHz(durationMs = 5000)
                _uiState.value = _uiState.value.copy(isRecording = false)

                if (pcmData != null) {
                    _uiState.value = _uiState.value.copy(isRecording = true) // will be cleared after
                    localAudioService.runPipeline(pcmData, visionFrames = emptyList())
                }

                // Small delay between captures
                delay(200)
            }
        }
    }

    fun stopListening() {
        _uiState.value = _uiState.value.copy(isRecording = false)
    }

    /**
     * Called by StreamViewModel.handleVideoFrame (throttled to ~1 fps).
     * Sends the latest vision frame along with the next audio chunk.
     */
    fun sendVideoFrameIfThrottled(bitmap: Bitmap) {
        if (!LocalAudioConfig.WHISPERX_URL.contains("localhost") && !LocalAudioConfig.WHISPERX_URL.contains("10.10")) return
        if (!sessionActive) return
        if (_uiState.value.connectionState != LiteLlmConnectionState.Ready) return

        val now = System.currentTimeMillis()
        if (now - lastVideoFrameTime < VIDEO_FRAME_INTERVAL_MS) return
        lastVideoFrameTime = now

        // Encode JPEG at 60% quality → base64
        val baos = java.io.ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 60, baos)
        val base64 = Base64.encodeToString(baos.toByteArray(), Base64.NO_WRAP)

        localAudioService.enqueueVisionFrame(base64)
    }

    fun clearError() {
        _uiState.value = _uiState.value.copy(errorMessage = null)
    }

    override fun onCleared() {
        super.onCleared()
        stopSession()
        localTtsManager.shutdown()
        localAudioService.close()
    }
}