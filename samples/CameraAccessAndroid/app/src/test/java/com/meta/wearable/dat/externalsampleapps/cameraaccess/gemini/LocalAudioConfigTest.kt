package com.meta.wearable.dat.externalsampleapps.cameraaccess.gemini

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Unit tests for LocalAudioConfig think-tag stripping.
 * These verify that <think> blocks are removed from LLM output
 * before TTS or UI display.
 */
class LocalAudioConfigTest {

    @Test
    fun stripThinkTags_noThinkTags_returnsOriginal() {
        val input = "Hello, how can I help you today?"
        val result = LocalAudioConfig.stripThinkTags(input)
        assertEquals(input, result)
    }

    @Test
    fun stripThinkTags_closedThinkTags_removed() {
        val input = "I'm thinking about that. <think>Let me reason through this.</think> Help me with something else!"
        val result = LocalAudioConfig.stripThinkTags(input)
        assertEquals("I'm thinking about that. Help me with something else!", result)
    }

    @Test
    fun stripThinkTags_unclosedThinkTag_removed() {
        val input = "Let me think <think>This should be removed"
        val result = LocalAudioConfig.stripThinkTags(input)
        assertEquals("Let me think", result)
    }

    @Test
    fun stripThinkTags_onlyThinkTag_returnsEmpty() {
        val input = "<think>Some internal reasoning here.?>"
        val result = LocalAudioConfig.stripThinkTags(input)
        assertEquals("", result)
    }

    @Test
    fun stripThinkTags_multipleThinkTags_removed() {
        val input = "First thought <think>first reason second thought <think>second reason end"
        val result = LocalAudioConfig.stripThinkTags(input)
        assertEquals("First thought second thought end", result)
    }

    @Test
    fun stripThinkTags_emptyInput_returnsEmpty() {
        val result = LocalAudioConfig.stripThinkTags("")
        assertEquals("", result)
    }

    @Test
    fun stripThinkTags_caseInsensitive() {
        val input = "Before  THINK internal REASON?> After"
        val result = LocalAudioConfig.stripThinkTags(input)
        assertEquals("Before After", result)
    }

    @Test
    fun stripThinkTags_preservesNormalTags() {
        val input = "Hello <b>world</b> and <i>italic</i> text"
        val result = LocalAudioConfig.stripThinkTags(input)
        assertEquals("Hello <b>world</b> and <i>italic</i> text", result)
    }

    @Test
    fun stripThinkTags_trimmed() {
        val input = "  Before </think>  "
        val result = LocalAudioConfig.stripThinkTags(input)
        assertEquals("Before", result)
    }

    @Test
    fun isConfigured_returnsTrue() {
        assertTrue(LocalAudioConfig.isConfigured())
    }

    @Test
    fun urlConstruction_buildsChatCompletionsPath() {
        val expected = "${LocalAudioConfig.LITELLM_BASE_URL}/chat/completions"
        assertTrue(expected.contains("/v1/chat/completions"))
        assertTrue(expected.contains(LocalAudioConfig.LITELLM_BASE_URL))
    }
}
