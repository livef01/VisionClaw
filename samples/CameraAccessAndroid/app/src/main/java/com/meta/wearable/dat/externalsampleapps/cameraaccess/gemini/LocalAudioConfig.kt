package com.meta.wearable.dat.externalsampleapps.cameraaccess.gemini

import android.content.Context
import com.meta.wearable.dat.externalsampleapps.cameraaccess.openclaw.OpenClawBridge

object LocalAudioConfig {
    // LiteLLM endpoint via OpenWebUI (OpenAI-compatible, single stable domain)
    const val LITELLM_BASE_URL = "https://openwebui-lxc.tail0b0f33.ts.net/api/v1"

    // OpenWebUI transcription endpoint (routes to WhisperX internally)
    const val WHISPERX_URL = "https://openwebui-lxc.tail0b0f33.ts.net/api/v1/audio/transcriptions"

    // OpenWebUI API key
    const val OPEN_WEB_UI_API_KEY = "sk-e2a4e9cd55784f95acefa809e202c7a9"

    // Android TTS — uses built-in engine via AudioTrack
    const val TTS_ENGINE = "com.google.android.tts"

    // OpenClaw gateway for tool calls / skill routing
    val openClawBridge: OpenClawBridge by lazy {
        OpenClawBridge()
    }

    fun isConfigured(): Boolean = true
}