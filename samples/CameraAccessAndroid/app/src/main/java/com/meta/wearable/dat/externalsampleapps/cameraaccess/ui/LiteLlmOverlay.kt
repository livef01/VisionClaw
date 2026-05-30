package com.meta.wearable.dat.externalsampleapps.cameraaccess.ui

import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.meta.wearable.dat.externalsampleapps.cameraaccess.gemini.LiteLlmConnectionState
import com.meta.wearable.dat.externalsampleapps.cameraaccess.gemini.LiteLlmUiState
import com.meta.wearable.dat.externalsampleapps.cameraaccess.openclaw.OpenClawConnectionState
import com.meta.wearable.dat.externalsampleapps.cameraaccess.openclaw.ToolCallStatus

@Composable
fun LiteLlmOverlay(
    uiState: LiteLlmUiState,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier.fillMaxWidth().padding(horizontal = 16.dp),
    ) {
        // Status bar
        LiteLlmStatusBar(
            connectionState = uiState.connectionState,
            openClawState = uiState.openClawConnectionState,
        )

        Spacer(modifier = Modifier.height(8.dp))

        // Transcripts
        if (uiState.userTranscript.isNotEmpty() || uiState.aiTranscript.isNotEmpty()) {
            LiteLlmTranscriptView(
                userTranscript = uiState.userTranscript,
                aiTranscript = uiState.aiTranscript,
            )
        }

        // Tool call status
        val toolStatus = uiState.toolCallStatus
        if (toolStatus !is ToolCallStatus.Idle) {
            Spacer(modifier = Modifier.height(4.dp))
            ToolCallStatusView(status = toolStatus)
        }

        // Speaking indicator
        if (uiState.isModelSpeaking) {
            Spacer(modifier = Modifier.height(4.dp))
            SpeakingIndicator()
        }
    }
}

@Composable
fun LiteLlmStatusBar(
    connectionState: LiteLlmConnectionState,
    openClawState: OpenClawConnectionState,
    modifier: Modifier = Modifier,
) {
    Row(
        modifier = modifier,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        StatusPill(
            label = "LLM",
            color = when (connectionState) {
                is LiteLlmConnectionState.Ready -> Color(0xFF4CAF50)
                is LiteLlmConnectionState.Connecting -> Color(0xFFFF9800)
                is LiteLlmConnectionState.Error -> Color(0xFFF44336)
                is LiteLlmConnectionState.Disconnected -> Color(0xFF9E9E9E)
            },
        )

        if (openClawState !is OpenClawConnectionState.NotConfigured) {
            StatusPill(
                label = "OpenClaw",
                color = when (openClawState) {
                    is OpenClawConnectionState.Connected -> Color(0xFF4CAF50)
                    is OpenClawConnectionState.Checking -> Color(0xFFFF9800)
                    is OpenClawConnectionState.Unreachable -> Color(0xFFF44336)
                    is OpenClawConnectionState.NotConfigured -> Color(0xFF9E9E9E)
                },
            )
        }
    }
}

@Composable
fun LiteLlmTranscriptView(
    userTranscript: String,
    aiTranscript: String,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier
            .background(Color.Black.copy(alpha = 0.5f), RoundedCornerShape(8.dp))
            .padding(horizontal = 12.dp, vertical = 8.dp),
    ) {
        if (userTranscript.isNotEmpty()) {
            Text(
                text = userTranscript,
                color = Color.White.copy(alpha = 0.6f),
                fontSize = 13.sp,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
            )
        }
        if (aiTranscript.isNotEmpty()) {
            Text(
                text = aiTranscript,
                color = Color.White,
                fontSize = 13.sp,
                maxLines = 3,
                overflow = TextOverflow.Ellipsis,
            )
        }
    }
}