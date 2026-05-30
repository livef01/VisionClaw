package com.meta.wearable.dat.externalsampleapps.cameraaccess.gemini

import android.content.Context
import android.graphics.Bitmap
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.meta.wearable.dat.externalsampleapps.cameraaccess.openclaw.GeminiFunctionCall
import com.meta.wearable.dat.externalsampleapps.cameraaccess.openclaw.GeminiToolCall
import com.meta.wearable.dat.externalsampleapps.cameraaccess.openclaw.GeminiToolCallCancellation
import com.meta.wearable.dat.externalsampleapps.cameraaccess.openclaw.OpenClawBridge
import com.meta.wearable.dat.externalsampleapps.cameraaccess.openclaw.OpenClawConnectionState
import com.meta.wearable.dat.externalsampleapps.cameraaccess.openclaw.OpenClawEventClient
import com.meta.wearable.dat.externalsampleapps.cameraaccess.openclaw.ToolCallRouter
import com.meta.wearable.dat.externalsampleapps.cameraaccess.openclaw.ToolCallStatus
import com.meta.wearable.dat.externalsampleapps.cameraaccess.settings.SettingsManager
import com.meta.wearable.dat.externalsampleapps.cameraaccess.stream.StreamingMode
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.util.Locale
import java.util.UUID

/**
 * Session ViewModel for the local AI pipeline.
 * Replaces GeminiSessionViewModel — same interface, different engine.
 *
 * Pipeline:
 *   AudioManager (mic) → LocalAudioService → WhisperX STT → LiteLLM → TTS → AudioManager (playback)
 *
 * Keeps: OpenClawBridge, ToolCallRouter, StreamViewModel wiring, WebRTC overlay hooks.
 */
class LiteLLMSessionViewModel(private val appContext: Context) : ViewModel() {

    companion object {
        private const val TAG = "LiteLLMSessionVM"
    }

    // ─── UI State (mirrors GeminiUiState) ────────────────────────────────────
    data class UiState(
        val isActive: Boolean = false,
        val connectionState: LocalAudioService.ConnectionState = LocalAudioService.ConnectionState.Disconnected,
        val isModelSpeaking: Boolean = false,
        val errorMessage: String? = null,
        val userTranscript: String = "",
        val aiTranscript: String = "",
        val toolCallStatus: ToolCallStatus = ToolCallStatus.Idle,
        val openClawConnectionState: OpenClawConnectionState = OpenClawConnectionState.NotConfigured,
    )

    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    // ─── Core components ────────────────────────────────────────────────────
    private val localAudioService = LocalAudioService(viewModelScope)
    private val openClawBridge = OpenClawBridge()
    private val eventClient = OpenClawEventClient()
    private val localAudioManager = LocalAudioManager()

    // Tool call router is created after OpenClaw check
    private var toolCallRouter: ToolCallRouter? = null

    // State observation job
    private var stateObservationJob: Job? = null

    // Last video frame time (for throttling)
    private var lastVideoFrameTime: Long = 0

    var streamingMode: StreamingMode = StreamingMode.GLASSES

    // ─── TTS (Android built-in) ────────────────────────────────────────────
    private var tts: TextToSpeech? = null
    private var isTtsInitialized = false

    init {
        initTts()
    }

    // ─── Session lifecycle ──────────────────────────────────────────────────

    fun startSession() {
        if (_uiState.value.isActive) return

        if (!LocalAudioConfig.isOpenClawConfigured) {
            _uiState.value = _uiState.value.copy(
                errorMessage = "OpenClaw not configured. Set host/token in Settings."
            )
            return
        }

        _uiState.value = _uiState.value.copy(isActive = true, errorMessage = null)

        viewModelScope.launch {
            // Wire audio callbacks
            localAudioManager.onAudioCaptured = lambda@{ data ->
                // Phone mode: mute mic while model speaks to prevent echo
                if (streamingMode == StreamingMode.PHONE && localAudioService.isModelSpeaking.value) {
                    return@lambda
                }
                localAudioService.sendAudio(data)
            }

            localAudioService.onAudioReceived = { data ->
                // Raw audio from pipeline → playback
                localAudioManager.playAudio(data)
            }

            localAudioService.onInterrupted = {
                localAudioManager.stopPlayback()
                tts?.stop()
            }

            localAudioService.onTurnComplete = {
                _uiState.value = _uiState.value.copy(userTranscript = "")
            }

            localAudioService.onInputTranscription = { text ->
                _uiState.value = _uiState.value.copy(
                    userTranscript = _uiState.value.userTranscript + text,
                    aiTranscript = ""
                )
            }

            localAudioService.onOutputTranscription = { text ->
                _uiState.value = _uiState.value.copy(
                    aiTranscript = _uiState.value.aiTranscript + text
                )
            }

            // TTS callback — receives LLM text and speaks it
            localAudioService.onResponseForTts = { text ->
                speak(text)
            }

            localAudioService.onDisconnected = { reason ->
                if (_uiState.value.isActive) {
                    stopSession()
                    _uiState.value = _uiState.value.copy(
                        errorMessage = "Connection lost: ${reason ?: "Unknown error"}"
                    )
                }
            }

            // Connect OpenClaw bridge and set up tool call routing
            openClawBridge.checkConnection()
            openClawBridge.resetSession()

            toolCallRouter = ToolCallRouter(openClawBridge, viewModelScope)

            // Handle tool calls from LLM → OpenClaw
            localAudioService.onToolCall = { toolCall ->
                for (call in toolCall.functionCalls) {
                    toolCallRouter?.handleToolCall(call) { response ->
                        localAudioService.sendToolResponse(response)
                        // After tool response, feed back to LLM for final answer
                        scope.launch {
                            // LocalAudioService handles re-chatting internally via conversationHistory
                        }
                    }
                }
            }

            localAudioService.onToolCallCancellation = { cancellation ->
                toolCallRouter?.cancelToolCalls(cancellation.ids)
            }

            // Observe service state
            stateObservationJob = viewModelScope.launch {
                while (isActive) {
                    delay(100)
                    _uiState.value = _uiState.value.copy(
                        connectionState = localAudioService.connectionState.value,
                        isModelSpeaking = localAudioService.isModelSpeaking.value,
                        toolCallStatus = openClawBridge.lastToolCallStatus.value,
                        openClawConnectionState = openClawBridge.connectionState.value,
                    )
                }
            }

            // Connect to LocalAudioService
            localAudioService.connect { setupOk ->
                if (!setupOk) {
                    val msg = when (val state = localAudioService.connectionState.value) {
                        is LocalAudioService.ConnectionState.Error -> state.message
                        else -> "Failed to connect to local AI stack"
                    }
                    _uiState.value = _uiState.value.copy(errorMessage = msg)
                    localAudioService.disconnect()
                    stateObservationJob?.cancel()
                    _uiState.value = _uiState.value.copy(
                        isActive = false,
                        connectionState = LocalAudioService.ConnectionState.Disconnected
                    )
                    return@connect
                }

                // Start mic capture
                try {
                    localAudioManager.startCapture()
                } catch (e: Exception) {
                    _uiState.value = _uiState.value.copy(
                        errorMessage = "Mic capture failed: ${e.message}"
                    )
                    localAudioService.disconnect()
                    stateObservationJob?.cancel()
                    _uiState.value = _uiState.value.copy(
                        isActive = false,
                        connectionState = LocalAudioService.ConnectionState.Disconnected
                    )
                }

                // Connect to OpenClaw event stream for proactive notifications
                if (SettingsManager.proactiveNotificationsEnabled) {
                    eventClient.onNotification = { text ->
                        val state = _uiState.value
                        if (state.isActive && localAudioService.connectionState.value == LocalAudioService.ConnectionState.Ready) {
                            localAudioService.sendTextMessage(text)
                        }
                    }
                    eventClient.connect()
                }
            }
        }
    }

    fun stopSession() {
        eventClient.disconnect()
        toolCallRouter?.cancelAll()
        toolCallRouter = null
        localAudioManager.stopCapture()
        localAudioService.disconnect()
        stateObservationJob?.cancel()
        stateObservationJob = null
        tts?.stop()
        _uiState.value = UiState()
    }

    fun sendVideoFrameIfThrottled(bitmap: Bitmap) {
        if (!SettingsManager.videoStreamingEnabled) return
        if (!_uiState.value.isActive) return
        if (localAudioService.connectionState.value != LocalAudioService.ConnectionState.Ready) return

        val now = System.currentTimeMillis()
        if (now - lastVideoFrameTime < LocalAudioConfig.VIDEO_FRAME_INTERVAL_MS) return
        lastVideoFrameTime = now

        localAudioService.sendVideoFrame(bitmap)
    }

    fun clearError() {
        _uiState.value = _uiState.value.copy(errorMessage = null)
    }

    // ─── TTS (Android built-in TextToSpeech) ───────────────────────────────

    private fun initTts() {
        tts = TextToSpeech(appContext, TextToSpeech.OnInitListener { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts?.let { tts ->
                    tts.language = Locale.US
                    tts.voice = tts.voices.firstOrNull { it.name.contains("en") } ?: tts.defaultVoice
                    tts.pitch = 1.0f
                    tts.speechRate = 1.0f
                    tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                        override fun onStart(utteranceId: String?) {
                            _uiState.value = _uiState.value.copy(isModelSpeaking = true)
                        }
                        override fun onDone(utteranceId: String?) {
                            _uiState.value = _uiState.value.copy(isModelSpeaking = false)
                        }
                        override fun onError(utteranceId: String?) {
                            _uiState.value = _uiState.value.copy(isModelSpeaking = false)
                        }
                    })
                    isTtsInitialized = true
                    Log.d(TAG, "Android TTS initialized")
                }
            } else {
                Log.e(TAG, "TTS init failed: $status")
            }
        })
    }

    private fun speak(text: String) {
        if (!isTtsInitialized || text.isBlank()) return
        val utteranceId = UUID.randomUUID().toString()
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, utteranceId)
        Log.d(TAG, "TTS speaking: ${text.take(50)}...")
    }

    override fun onCleared() {
        super.onCleared()
        stopSession()
        tts?.shutdown()
        tts = null
    }
}