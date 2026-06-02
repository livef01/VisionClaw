package com.meta.wearable.dat.externalsampleapps.cameraaccess.gemini

/**
 * Endpoints and shared constants for the local audio pipeline
 * (WhisperX STT → LiteLLM → Android TTS). The actual `OpenClawBridge` instance
 * is owned by the ViewModel that needs it; this object is purely config.
 */
object LocalAudioConfig {
    // LiteLLM endpoint (OpenAI-compatible)
    const val LITELLM_BASE_URL = "http://10.10.10.150:18790/v1"

    // WhisperX transcription server
    const val WHISPERX_URL = "http://10.10.10.150:5000/transcribe"

    // Android TTS — uses built-in engine via AudioTrack
    const val TTS_ENGINE = "com.google.android.tts"

    fun isConfigured(): Boolean = true
}
