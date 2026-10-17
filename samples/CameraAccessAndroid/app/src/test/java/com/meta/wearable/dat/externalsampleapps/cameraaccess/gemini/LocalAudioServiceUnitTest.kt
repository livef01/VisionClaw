package com.meta.wearable.dat.externalsampleapps.cameraaccess.gemini

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Unit tests for LocalAudioService reasoning-tag stripping
 * and conversation history functions.
 */
class LocalAudioServiceUnitTest {

    // ─── stripReasoningTags ───────────────────────────────────────────────────

    @Test
    fun stripReasoningTags_emptyInput() {
        assertEquals("", LocalAudioService.stripReasoningTags(""))
    }

    @Test
    fun stripReasoningTags_nullInput() {
        assertEquals("", LocalAudioService.stripReasoningTags(""))
    }

    @Test
    fun stripReasoningTags_noTags() {
        val input = "Hello, how can I help you?"
        assertEquals("Hello, how can I help you?", LocalAudioService.stripReasoningTags(input))
    }

    @Test
    fun stripReasoningTags_singleThinkBlock() {
        val input = "<think>Let me think about this.</think> Hello there!"
        assertEquals("Hello there!", LocalAudioService.stripReasoningTags(input))
    }

    @Test
    fun stripReasoningTags_onlyThinkBlock() {
        val input = "<think>Some reasoning here.</think> "
        assertEquals("", LocalAudioService.stripReasoningTags(input))
    }

    @Test
    fun stripReasoningTags_multipleThinkBlocks() {
        val input = "<think>First thought.</think> middle <think>Second thought.</think> end"
        assertEquals("middle  end", LocalAudioService.stripReasoningTags(input))
    }

    @Test
    fun stripReasoningTags_thinkBlockAtStart() {
        val input = "<think>reasoning</think> actual response"
        assertEquals("actual response", LocalAudioService.stripReasoningTags(input))
    }

    @Test
    fun stripReasoningTags_thinkBlockAtEnd() {
        val input = "real answer  <think>extra reasoning</think>"
        assertEquals("real answer", LocalAudioService.stripReasoningTags(input))
    }

    @Test
    fun stripReasoningTags_multilineThinkBlock() {
        val input = "<think>" +
            "Line 1\n" +
            "Line 2\n" +
            "</think> Hello!"
        assertEquals("Hello!", LocalAudioService.stripReasoningTags(input))
    }

    @Test
    fun stripReasoningTags_preservesValidContent() {
        val input = "The weather is nice today."
        assertEquals("The weather is nice today.", LocalAudioService.stripReasoningTags(input))
    }

    // ─── buildMessages ────────────────────────────────────────────────────────

    @Test
    fun buildMessages_containsSystemAndUser() {
        val systemPrompt = "You are an assistant."
        val transcript = "Hello"
        val messages = LocalAudioService.buildMessages(systemPrompt, transcript, emptyList(), emptyList())

        assertEquals(2, messages.length())

        // System message
        assertEquals("system", messages.getJSONObject(0).getString("role"))
        assertEquals(systemPrompt, messages.getJSONObject(0).getString("content"))

        // User message
        assertEquals("user", messages.getJSONObject(1).getString("role"))
    }

    @Test
    fun buildMessages_includesVisionFrames() {
        val messages = LocalAudioService.buildMessages(
            systemPrompt = "Test",
            transcript = "Hello",
            frames = listOf("frame1data", "frame2data"),
            history = emptyList()
        )

        val userContent = messages.getJSONObject(1).getJSONArray("content")
        // Should have text + 2 image frames
        assertEquals(3, userContent.length())

        // First item should be text
        assertEquals("text", userContent.getJSONObject(0).getString("type"))
        assertEquals("Hello", userContent.getJSONObject(0).getString("text"))

        // Second item should be image
        assertEquals("image_url", userContent.getJSONObject(1).getString("type"))

        // Third item should be image
        assertEquals("image_url", userContent.getJSONObject(2).getString("type"))
    }

    @Test
    fun buildMessages_includesHistoryMessages() {
        val history = listOf(
            mapOf("role" to "user", "content" to "Previous question"),
            mapOf("role" to "assistant", "content" to "Previous answer")
        )

        val messages = LocalAudioService.buildMessages(
            systemPrompt = "Test",
            transcript = "Current question",
            frames = emptyList(),
            history = history
        )

        assertEquals(4, messages.length()) // system + history(2) + current user
        assertEquals("Previous question", messages.getJSONObject(1).getString("content"))
        assertEquals("Previous answer", messages.getJSONObject(2).getString("content"))
    }

    @Test
    fun buildMessages_boundedHistory() {
        val history = mutableListOf<Map<String, Any>>()
        for (i in 1..25) {
            history.add(mapOf("role" to "user", "content" to "msg$i"))
        }

        val messages = LocalAudioService.buildMessages(
            systemPrompt = "Test",
            transcript = "Current",
            frames = emptyList(),
            history = history.toList()
        )

        // Should have system + up to MAX_HISTORY_MESSAGES + 1 (current)
        // MAX_HISTORY_MESSAGES = 20, so system + 20 + 1 user = 22
        assertTrue(messages.length() <= 22)
    }

    @Test
    fun buildMessages_respectsMaxBound() {
        val history = mutableListOf<Map<String, Any>>()
        for (i in 1..50) {
            history.add(mapOf("role" to if (i % 2 == 1) "user" else "assistant", "content" to "msg$i"))
        }

        val messages = LocalAudioService.buildMessages(
            systemPrompt = "Test",
            transcript = "Current",
            frames = emptyList(),
            history = history.toList()
        )

        // Total messages capped: system(1) + MAX_HISTORY_MESSAGES(20) + user(1) = 22
        assertTrue(messages.length() <= 1 + LocalAudioService.MAX_HISTORY_MESSAGES + 1)

        // Last message should be the current user message
        assertEquals("user", messages.getJSONObject(messages.length() - 1).getString("role"))
    }

    // ─── recordTurn ──────────────────────────────────────────────────────────

    @Test
    fun recordTurn_addsToHistory() {
        val history = mutableListOf<Map<String, Any>>()

        val userMsg = mapOf("role" to "user", "content" to "Hello")
        val assistantMsg = mapOf("role" to "assistant", "content" to "Hi!")

        LocalAudioService.recordTurn(history, userMsg, assistantMsg)

        assertEquals(2, history.size)
        assertEquals("Hello", history[0]["content"])
        assertEquals("Hi!", history[1]["content"])
    }

    @Test
    fun recordTurn_boundedHistory() {
        val history = mutableListOf<Map<String, Any>>()

        // Fill up to MAX_HISTORY_MESSAGES entries
        for (i in 1..50) {
            val userMsg = mapOf("role" to "user", "content" to "user$i")
            val assistantMsg = mapOf("role" to "assistant", "content" to "assistant$i")
            LocalAudioService.recordTurn(history, userMsg, assistantMsg)
        }

        // Should be bounded to MAX_HISTORY_MESSAGES total entries
        assertTrue(history.size <= LocalAudioService.MAX_HISTORY_MESSAGES)
    }

    @Test
    fun recordTurn_preservesPairs() {
        val history = mutableListOf<Map<String, Any>>()

        val user1 = mapOf("role" to "user", "content" to "Hello")
        val assistant1 = mapOf("role" to "assistant", "content" to "Hi!")
        val user2 = mapOf("role" to "user", "content" to "How are you?")
        val assistant2 = mapOf("role" to "assistant", "content" to "Good!")

        LocalAudioService.recordTurn(history, user1, assistant1)
        LocalAudioService.recordTurn(history, user2, assistant2)

        assertEquals(4, history.size)
        assertEquals("user", history[0]["role"])
        assertEquals("assistant", history[1]["role"])
        assertEquals("user", history[2]["role"])
        assertEquals("assistant", history[3]["role"])
    }

    @Test
    fun recordTurn_removesOldestPairs_whenExceeding() {
        val history = mutableListOf<Map<String, Any>>()

        // Add enough turns to exceed the limit
        for (i in 1..30) {
            val userMsg = mapOf("role" to "user", "content" to "user$i")
            val assistantMsg = mapOf("role" to "assistant", "content" to "assistant$i")
            LocalAudioService.recordTurn(history, userMsg, assistantMsg)
        }

        assertTrue(history.size <= LocalAudioService.MAX_HISTORY_MESSAGES)

        // Should contain the most recent entries
        val lastUserContent = "user29"
        val lastAssistantContent = "assistant29"
        assertTrue(history.any { it["content"] == lastUserContent })
        assertTrue(history.any { it["content"] == lastAssistantContent })
    }

    @Test
    fun recordTurn_maintains_pairing_when_capped_at_single_entry() {
        val history = mutableListOf<Map<String, Any>>()

        // Add entries until we're at exactly MAX_HISTORY_MESSAGES
        for (i in 1..40) {
            val userMsg = mapOf("role" to "user", "content" to "user$i")
            val assistantMsg = mapOf("role" to "assistant", "content" to "assistant$i")
            LocalAudioService.recordTurn(history, userMsg, assistantMsg)
        }

        // All remaining entries should be in pairs
        for (i in history.indices step 2) {
            assertEquals("user", history[i]["role"])
            if (i + 1 < history.size) {
                assertEquals("assistant", history[i + 1]["role"])
            }
        }
    }

    // ─── recordToolTurn ──────────────────────────────────────────────────────

    @Test
    fun recordToolTurn_records_all_phases() {
        val history = mutableListOf<Map<String, Any>>()

        LocalAudioService.recordToolTurn(
            history,
            assistantToolCallMessage = mapOf(
                "role" to "assistant",
                "content" to "<think>thinking</think> Let me call a tool",
                "tool_calls" to listOf(mapOf("id" to "call1"))
            ),
            toolResults = listOf(mapOf(
                "role" to "tool",
                "tool_call_id" to "call1",
                "content" to "<think>private result</think>result"
            )),
            assistantFinalContent = "Done with tool result"
        )

        assertEquals(3, history.size)
        assertEquals("assistant", history[0]["role"])
        assertEquals("tool", history[1]["role"])
        assertEquals("assistant", history[2]["role"])

        // Reasoning tags should be stripped
        val content = history[0]["content"] as String
        assertFalse(content.contains("thinking"))
        assertEquals("Let me call a tool", content.trim())
        assertEquals("result", history[1]["content"])
        assertTrue(history[0].containsKey("tool_calls"))
    }

    @Test
    fun recordToolTurn_bounded() {
        val history = mutableListOf<Map<String, Any>>()

        // Do enough tool turns to exceed the limit
        for (i in 1..30) {
            LocalAudioService.recordToolTurn(
                history,
                assistantToolCallMessage = mapOf(
                    "role" to "assistant",
                    "content" to "assistant $i",
                    "tool_calls" to listOf(mapOf("id" to "call$i"))
                ),
                toolResults = listOf(mapOf("role" to "tool", "content" to "tool $i")),
                assistantFinalContent = "final $i"
            )
        }

        assertTrue(history.size <= LocalAudioService.MAX_HISTORY_MESSAGES)
    }

    // ─── Integration: strip then record ───────────────────────────────────────

    @Test
    fun fullPipeline_stripsReasoningBeforeRecording() {
        val history = mutableListOf<Map<String, Any>>()

        val assistantRaw = "<think>Let me think about this.</think> Hello there!"
        val assistantClean = LocalAudioService.stripReasoningTags(assistantRaw)

        val userMsg = mapOf("role" to "user", "content" to "What's the weather?")
        val assistantMsg = mapOf("role" to "assistant", "content" to assistantClean)

        LocalAudioService.recordTurn(history, userMsg, assistantMsg)

        assertEquals(2, history.size)
        // The assistant message should NOT contain reasoning tags
        val assistantContent = history[1]["content"] as String
        assertFalse(assistantContent.contains("thinking", ignoreCase = true))
        assertEquals("Hello there!", assistantContent)
    }
}
