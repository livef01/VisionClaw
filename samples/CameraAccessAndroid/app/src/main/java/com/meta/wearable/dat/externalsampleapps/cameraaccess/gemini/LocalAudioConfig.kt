package com.meta.wearable.dat.externalsampleapps.cameraaccess.gemini

import com.meta.wearable.dat.externalsampleapps.cameraaccess.settings.SettingsManager

/**
 * Local AI Stack Configuration — replaces GeminiConfig for abacus-glasses.
 *
 * Routes to:
 * - LiteLLM (10.10.10.150:4000) — chat completions, vision
 * - WhisperX (10.10.10.150:9000) — STT
 * - OpenHue / local TTS — TTS playback
 */
object LocalAudioConfig {

    // ─── LiteLLM (LLM + Vision) ─────────────────────────────────────────────
    const val LITELLM_BASE_URL = "http://10.10.10.150:4000"

    const val MODEL = "qwen/qwen2-vl-7b-instruct"  // vision model via LiteLLM
    const val MODEL_VISION = "qwen/qwen2-vl-7b-instruct"

    // Fallback: text-only
    const val MODEL_TEXT = "qwen/qwen2.5-32b-instruct"

    // ─── WhisperX (STT) ─────────────────────────────────────────────────────
    const val WHISPER_URL = "http://10.10.10.150:9000"
    const val WHISPER_API_PATH = "/api/v1/transcribe"

    // ─── TTS ─────────────────────────────────────────────────────────────────
    // Android TTS — no external server needed (fast, offline)
    // Future: Coqui/RVC at:
    // const val TTS_URL = "http://10.10.10.150:5000"

    // ─── Audio parameters ────────────────────────────────────────────────────
    const val INPUT_AUDIO_SAMPLE_RATE = 16000
    const val OUTPUT_AUDIO_SAMPLE_RATE = 24000
    const val AUDIO_CHANNELS = 1
    const val AUDIO_BITS_PER_SAMPLE = 16

    // Minimum bytes before triggering STT (~100ms at 16kHz mono PCM16)
    const val MIN_AUDIO_CHUNK_BYTES = 3200

    // ─── Video streaming ────────────────────────────────────────────────────
    const val VIDEO_FRAME_INTERVAL_MS = 1000L
    const val VIDEO_JPEG_QUALITY = 50

    // ─── OpenClaw (tool calls) ───────────────────────────────────────────────
    val openClawHost: String
        get() = SettingsManager.openClawHost

    val openClawPort: Int
        get() = SettingsManager.openClawPort

    val openClawGatewayToken: String
        get() = SettingsManager.openClawGatewayToken

    val isOpenClawConfigured: Boolean
        get() = openClawGatewayToken != "YOUR_OPENCLAW_GATEWAY_TOKEN"
                && openClawGatewayToken.isNotEmpty()
                && openClawHost != "http://YOUR_MAC_HOSTNAME.local"

    // ─── System instruction ──────────────────────────────────────────────────
    val systemInstruction: String
        get() = SettingsManager.geminiSystemPrompt

    // ─── Derived helpers ─────────────────────────────────────────────────────
    val whisperTranscribeUrl: String
        get() = "$WHISPER_URL$WHISPER_API_PATH"

    val litellmChatUrl: String
        get() = "$LITELLM_BASE_URL/v1/chat/completions"
}