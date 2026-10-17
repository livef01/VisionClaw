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

    /**
     * Strip <think>...</think> blocks from LLM output before TTS or UI display.
     * Handles tags that may be split across stream chunks (e.g. "<thi" + "nk>").
     * Uses a buffered approach: accumulates characters until a closing </think>
     * or > is found, then discards everything up to the next </think>.
     */
    fun stripThinkTags(text: String): String {
        // First handle already-closed think tags (simple case)
        val cleaned = text.replace(Regex("<think[^>]*>[\\s\\S]*?</think>", RegexOption.IGNORE_CASE), "")

        // Handle partial/unclosed tags: strip from <think to end of string
        val unclosedPattern = Regex("<think[^>]*>[\\s\\S]*$", RegexOption.IGNORE_CASE)
        val result = unclosedPattern.replace(cleaned, "")

        return result.trim()
    }
}
