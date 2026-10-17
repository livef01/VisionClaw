package com.meta.wearable.dat.externalsampleapps.cameraaccess.gemini

/**
 * Endpoints and shared constants for the local audio pipeline
 * (WhisperX STT → LiteLLM → Android TTS). The actual `OpenClawBridge` instance
 * is owned by the ViewModel that needs it; this object is purely config.
 */
object LocalAudioConfig {
    // Working LiteLLM proxy endpoint (OpenAI-compatible)
    const val LITELLM_BASE_URL = "http://100.105.130.81:4000/v1"

    // WhisperX transcription server
    const val WHISPERX_URL = "http://10.10.10.150:5000/transcribe"

    // Android TTS — uses built-in engine via AudioTrack
    const val TTS_ENGINE = "com.google.android.tts"

    fun isConfigured(): Boolean = true

    /**
     * Strip <think>...</think> blocks from LLM output before TTS or UI display.
     * Handles already-closed tags and partial/unclosed tags (e.g. LLM streams
     * that cut off mid-think or output just "<think").  Does NOT buffer across
     * chunks — the caller must provide the complete string or use
     * processStreamChunk() for incremental streaming.
     */
    fun stripThinkTags(text: String): String {
        val cleaned = text.replace(Regex("<think[^>]*>[\\s\\S]*?</think>", RegexOption.IGNORE_CASE), "")

        val unclosedPattern = Regex("<think[^>]*>[\\s\\S]*$", RegexOption.IGNORE_CASE)
        val result = unclosedPattern.replace(cleaned, "")

        return result.trim()
    }

    /**
     * Process a single stream chunk, returning the non-think portion.
     * Maintains minimal internal state so a partial opening tag such as
     * "<thi" that arrives before "<nk>" is correctly discarded.
     *
     * Returns Pair<textBeforeThinkTag, leftoverPartialTagOrNull>.
     */
    fun processStreamChunk(chunk: String): Pair<String, String?> {
        var out = ""
        var remaining = chunk

        if (_inThinkTag) {
            val closeIdx = remaining.indexOfIgnoreCase("?>")
            if (closeIdx >= 0) {
                _inThinkTag = false
                remaining = remaining.substring(closeIdx + 2)
            } else {
                return Pair("", null)
            }
        }

        var idx = 0
        while (idx < remaining.length) {
            val openStart = remaining.indexOf("<think", idx, ignoreCase = true)
            if (openStart < 0) {
                out += remaining.substring(idx)
                break
            }

            out += remaining.substring(idx, openStart)

            val afterOpenStart = openStart + 6
            val openEnd = remaining.indexOf(">", afterOpenStart)
            if (openEnd < 0) {
                // Opening tag not completed yet (e.g. "<thi" arrived alone)
                _inThinkTag = true
                _partialTag = remaining.substring(openStart)
                break
            }

            val afterOpen = openEnd + 1
            val closeIdx = remaining.indexOfIgnoreCase("?>", afterOpen)
            if (closeIdx >= 0) {
                idx = closeIdx + 2
            } else {
                _inThinkTag = true
                _partialTag = remaining.substring(afterOpen)
                break
            }
        }

        return Pair(out.trim(), if (_inThinkTag) {
            if (_partialTag != null && _partialTag!!.contains(">")) null else _partialTag
        } else null)
    }

    private var _inThinkTag = false
    private var _partialTag: String? = null

    private fun String.indexOfIgnoreCase(substring: String, startIndex: Int = 0): Int {
        val rest = this.subSequence(startIndex, this.length).toString().lowercase()
        val search = substring.lowercase()
        val found = rest.indexOf(search)
        return if (found >= 0) startIndex + found else -1
    }

    fun resetStreamState() {
        _inThinkTag = false
        _partialTag = null
    }
}
