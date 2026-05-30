package com.meta.wearable.dat.externalsampleapps.cameraaccess.gemini

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import androidx.core.content.ContextCompat
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream

class LocalAudioManager(private val context: Context) {

    enum class State { IDLE, RECORDING, PLAYING }
    private val _state = MutableStateFlow(State.IDLE)
    val state: StateFlow<State> = _state

    private var audioRecord: AudioRecord? = null
    private var audioTrack: AudioTrack? = null

    private val sampleRate = 16000
    private val channelConfig = AudioFormat.CHANNEL_IN_MONO
    private val audioFormat = AudioFormat.ENCODING_PCM_16BIT
    private val bufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)

    // --- Mic capture ---
    fun hasRecordPermission(): Boolean =
        ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED

    suspend fun capturePcm16kHz(durationMs: Int = 5000): ByteArray? = withContext(Dispatchers.IO) {
        if (!hasRecordPermission()) return@withContext null
        _state.value = State.RECORDING

        val record = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            channelConfig,
            audioFormat,
            bufferSize
        )
        audioRecord = record

        val buffer = ByteArrayOutputStream()
        val bytesPerSample = 2
        val totalSamples = (sampleRate * durationMs / 1000)
        val totalBytes = totalSamples * bytesPerSample
        val totalFrames = totalBytes / bufferSize

        try {
            record.startRecording()
            val buf = ByteArray(bufferSize)

            for (i in 0 until totalFrames) {
                val read = record.read(buf, 0, bufferSize)
                if (read > 0) buffer.write(buf, 0, read)
            }
        } finally {
            record.stop()
            record.release()
            audioRecord = null
            _state.value = State.IDLE
        }

        buffer.toByteArray()
    }

    // --- TTS playback via AudioTrack ---
    fun playPcm16kHz(pcmData: ByteArray) {
        _state.value = State.PLAYING

        val track = AudioTrack.Builder()
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(audioFormat)
                    .setSampleRate(sampleRate)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build()
            )
            .setBufferSizeInBytes(pcmData.size)
            .setTransferMode(AudioTrack.MODE_STATIC)
            .build()

        audioTrack = track

        try {
            track.write(pcmData, 0, pcmData.size)
            track.play()
        } finally {
            // Let playback finish naturally; caller can await completion
            _state.value = State.IDLE
        }
    }

    fun stopPlayback() {
        audioTrack?.apply {
            stop()
            release()
        }
        audioTrack = null
        _state.value = State.IDLE
    }
}
