package com.meta.wearable.dat.externalsampleapps.cameraaccess.gemini

import android.content.Context
import com.meta.wearable.dat.externalsampleapps.cameraaccess.openclaw.OpenClawBridge

object LocalAudioConfig {
    // LiteLLM endpoint (OpenAI-compatible)
    const val LITELLM_BASE_URL = "https://ads-opposition-proud-bush.trycloudflare.com/v1"

    // WhisperX transcription server
    const val WHISPERX_URL = "https://mineral-progressive-improve-theorem.trycloudflare.com"

    // Android TTS — uses built-in engine via AudioTrack
    const val TTS_ENGINE = "com.google.android.tts"

    // OpenClaw gateway for tool calls / skill routing
    val openClawBridge: OpenClawBridge by lazy {
        OpenClawBridge()
    }

    fun isConfigured(): Boolean = true
}
