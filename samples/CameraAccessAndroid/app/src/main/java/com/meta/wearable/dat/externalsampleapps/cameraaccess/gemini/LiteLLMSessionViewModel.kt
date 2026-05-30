package com.meta.wearable.dat.externalsampleapps.cameraaccess.gemini

import android.app.Application
import android.util.Base64
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

data class LiteLlmUiState(
    val transcript: String = "",
    val llmResponse: String = "",
    val isRecording: Boolean = false,
    val ttsReady: Boolean = false,
    val error: String? = null,
    val isUsingLocalPipeline: Boolean = true
)

class LiteLLMSessionViewModel(application: Application) : AndroidViewModel(application) {

    private val localAudioManager = LocalAudioManager(application)
    private val localTtsManager = LocalTtsManager(application)
    private val localAudioService = LocalAudioService()

    private val _uiState = MutableStateFlow(LiteLlmUiState())
    val uiState: StateFlow<LiteLlmUiState> = _uiState

    init {
        localTtsManager.init()

        // Wire TTS callback into the pipeline
        localAudioService.onResponseForTts = { text ->
            localTtsManager.speak(text)
            _uiState.value = _uiState.value.copy(llmResponse = text)
        }

        // Observe service events
        viewModelScope.launch {
            localAudioService.events.collect { event ->
                when (event) {
                    is LocalAudioService.PipelineEvent.Transcript -> {
                        _uiState.value = _uiState.value.copy(transcript = event.text)
                    }
                    is LocalAudioService.PipelineEvent.LlmResponse -> {
                        _uiState.value = _uiState.value.copy(llmResponse = event.text)
                    }
                    is LocalAudioService.PipelineEvent.TtsReady -> {
                        localAudioManager.playPcm16kHz(event.pcmData)
                    }
                    is LocalAudioService.PipelineEvent.Error -> {
                        _uiState.value = _uiState.value.copy(error = event.message)
                    }
                    null -> {}
                }
            }
        }
    }

    fun startRecording() {
        if (_uiState.value.isRecording) return
        _uiState.value = _uiState.value.copy(isRecording = true, error = null)

        viewModelScope.launch {
            val pcmData = localAudioManager.capturePcm16kHz(durationMs = 5000)
            _uiState.value = _uiState.value.copy(isRecording = false)

            if (pcmData != null) {
                // Run full pipeline: STT → LLM → TTS
                localAudioService.runPipeline(pcmData)
            } else {
                _uiState.value = _uiState.value.copy(error = "Mic capture failed")
            }
        }
    }

    fun stopRecording() {
        // Recording auto-stops after duration; expose for manual stop if needed
    }

    fun clearTranscript() {
        _uiState.value = _uiState.value.copy(transcript = "", llmResponse = "", error = null)
    }

    override fun onCleared() {
        super.onCleared()
        localTtsManager.shutdown()
        localAudioService.close()
    }
}