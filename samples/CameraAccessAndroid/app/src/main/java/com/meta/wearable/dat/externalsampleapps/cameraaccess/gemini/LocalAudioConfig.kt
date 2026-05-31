package com.meta.wearable.dat.externalsampleapps.cameraaccess.gemini

import android.content.Context
import com.meta.wearable.dat.externalsampleapps.cameraaccess.openclaw.OpenClawBridge

object LocalAudioConfig {
    // LiteLLM endpoint (OpenAI-compatible) — VM 451
    const val LITELLM_BASE_URL = "http://100.105.130.81:4000/v1"

    // WhisperX transcription server — VM 451
    const val WHISPERX_URL = "http://100.105.130.81:5000/transcribe_base64"

    // Android TTS — uses built-in engine via AudioTrack
    const val TTS_ENGINE = "com.google.android.tts"

    // OpenClaw gateway for tool calls / skill routing
    val openClawBridge: OpenClawBridge by lazy {
        OpenClawBridge()
    }

    fun isConfigured(): Boolean = true
}