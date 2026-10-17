package com.meta.wearable.dat.externalsampleapps.cameraaccess.gemini

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LocalAudioConfigTest {

    @Test
    fun stripThinkTags_noThinkTags_returnsOriginal() {
        val input = "Hello, how can I help you today?"
        val result = LocalAudioConfig.stripThinkTags(input)
        assertEquals(input, result)
    }

    @Test
    fun stripThinkTags_closedThinkTags_removed() {
        val input = "Start <think>inner text</think> end"
        val result = LocalAudioConfig.stripThinkTags(input)
        assertEquals("Start end", result)
    }

    @Test
    fun stripThinkTags_unclosedThinkTag_removed() {
        val input = "Before <think>after"
        val result = LocalAudioConfig.stripThinkTags(input)
        assertEquals("Before", result)
    }

    @Test
    fun stripThinkTags_onlyThinkTag_returnsEmpty() {
        val input = "inner text?>"
        val result = LocalAudioConfig.stripThinkTags(input)
        assertEquals("", result)
    }

    @Test
    fun stripThinkTags_multipleThinkTags_removed() {
        val input = "A thought1  thought2  thought3 end"
        val result = LocalAudioConfig.stripThinkTags(input)
        assertEquals("A end", result)
    }

    @Test
    fun stripThinkTags_emptyInput_returnsEmpty() {
        val result = LocalAudioConfig.stripThinkTags("")
        assertEquals("", result)
    }

    @Test
    fun stripThinkTags_preservesNormalTags() {
        val input = "Hello <b>world</b> text"
        val result = LocalAudioConfig.stripThinkTags(input)
        assertEquals("Hello <b>world</b> text", result)
    }

    @Test
    fun stripThinkTags_trimmed() {
        val input = "  Before ?>"
        val result = LocalAudioConfig.stripThinkTags(input)
        assertEquals("Before", result)
    }

    // ── Stream-safe helpers ────────────────────────────────────────────────

    @Test
    fun processStreamChunk_completeThinkTagInChunk_returnsClearedText() {
        val chunk = "Hello  inner  end "
        val (out, leftover) = LocalAudioConfig.processStreamChunk(chunk)
        assertEquals("Hello ", out)
        assertEquals("", leftover ?: "")
    }

    @Test
    fun processStreamChunk_partialThinkTag_savedForNextChunk() {
        val chunk = "text  part1"
        var leftover: String?
        LocalAudioConfig.processStreamChunk(chunk)
        leftover = LocalAudioConfig._partialTag
        assertTrue(leftover != null && leftover!!.contains("part1"))
    }

    @Test
    fun processStreamChunk_noThinkTag_passesThrough() {
        val chunk = "Hello world"
        val (out, _) = LocalAudioConfig.processStreamChunk(chunk)
        assertEquals("Hello world", out)
    }

    @Test
    fun isConfigured_returnsTrue() {
        assertTrue(LocalAudioConfig.isConfigured())
    }

    @Test
    fun endpoint_isWorkingProxy() {
        assertEquals("http://100.105.130.81:4000/v1", LocalAudioConfig.LITELLM_BASE_URL)
    }

    @Test
    fun urlConstruction_buildsChatCompletionsPath() {
        val expected = "http://100.105.130.81:4000/v1" + "/chat/completions"
        assertTrue(expected.contains("/v1/chat/completions"))
    }
}
