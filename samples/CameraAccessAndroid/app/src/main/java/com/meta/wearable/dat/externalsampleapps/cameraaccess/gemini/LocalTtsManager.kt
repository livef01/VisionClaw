package com.meta.wearable.dat.externalsampleapps.cameraaccess.gemini

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioManager
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import java.util.Locale
import java.util.UUID

/**
 * Local TTS Manager — wraps Android TextToSpeech.
 *
 * Replaces Coqui/RVC TTS from the plan with Android's built-in TTS.
 * Fast, offline, no external server needed.
 */
class LocalTtsManager(private val context: Context) : TextToSpeech.OnInitListener {

    companion object {
        private const val TAG = "LocalTtsManager"
    }

    private var tts: TextToSpeech? = null
    private var isInitialized = false
    private var currentUtteranceId: String? = null

    var onSpeakingComplete: (() -> Unit)? = null
    var onError: ((String) -> Unit)? = null

    // Audio focus for TTS playback through phone speaker
    private val audioFocusRequest = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT)
        .setAudioAttributes(
            AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                .build()
        )
        .setOnAudioFocusChangeListener { /* ignore focus changes */ }
        .build()

    init {
        val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        val result = audioManager.requestAudioFocus(audioFocusRequest)
        if (result == AudioManager.AUDIOFOCUS_REQUEST_GRANTED) {
            tts = TextToSpeech(context, this)
        } else {
            // Try anyway
            tts = TextToSpeech(context, this)
        }
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts?.let { tts ->
                tts.language = Locale.US
                tts.voice = tts.voices.firstOrNull { it.name.contains("en") } ?: tts.defaultVoice
                tts.pitch = 1.0f
                tts.speechRate = 1.0f
                tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                    override fun onStart(utteranceId: String?) {
                        currentUtteranceId = utteranceId
                    }
                    override fun onDone(utteranceId: String?) {
                        currentUtteranceId?.let { id ->
                            if (id == utteranceId) {
                                onSpeakingComplete?.invoke()
                            }
                        }
                    }
                    override fun onError(utteranceId: String?) {
                        onError?.invoke("TTS error for utterance: $utteranceId")
                    }
                })
                isInitialized = true
                Log.d(TAG, "Android TTS initialized")
            }
        } else {
            Log.e(TAG, "TTS initialization failed: $status")
        }
    }

    fun speak(text: String) {
        if (!isInitialized) {
            Log.w(TAG, "TTS not initialized, skipping speak")
            return
        }
        val utteranceId = UUID.randomUUID().toString()
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, utteranceId)
        Log.d(TAG, "Speaking: ${text.take(50)}...")
    }

    fun stop() {
        tts?.stop()
        audioFocusRequest.abandon()
        Log.d(TAG, "TTS stopped")
    }

    fun isCurrentlySpeaking(): Boolean {
        return tts?.isSpeaking ?: false
    }

    fun shutdown() {
        tts?.shutdown()
        tts = null
        isInitialized = false
    }
}