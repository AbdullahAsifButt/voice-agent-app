import streamlit as st
import os
import asyncio
import edge_tts
from groq import Groq
from dotenv import load_dotenv

# Load keys
load_dotenv()

# Setup Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# 1. THE "EARS" (Web Version)
# In Streamlit, we don't "record" in a loop. We wait for the user to upload voice.
def transribe_audio(audio_bytes):
    try:
        return client.audio.transcriptions.create(
            file=("input.wav", audio_bytes), model="whisper-large-v3", language="en"
        ).text
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        return None


# 2. THE "BRAIN" (Groq Llama 3)
def get_ai_response(text):
    try:
        completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful voice assistant. Keep answers short (1-2 sentences).",
                },
                {"role": "user", "content": text},
            ],
            model="llama-3.3-70b-versatile",
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"AI Error: {e}")
        return None


# 3. THE "MOUTH" (Edge-TTS - Free & Unlimited)
async def generate_audio(text, output_file="response.mp3"):
    # "en-US-AriaNeural" is a high-quality free voice from Microsoft
    communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
    await communicate.save(output_file)
    return output_file


# --- MAIN APP UI ---
# ... (Keep imports and helper functions the same) ...


# --- MAIN APP UI ---
def main():
    st.title("Real-Time Voice Agent")

    # 1. Session State Setup (The "Memory" Fix)
    # We create a counter to track the widget's ID
    if "voice_key" not in st.session_state:
        st.session_state.voice_key = 0

    # 2. Callback function to reset the key
    def reset_audio():
        st.session_state.voice_key += 1

    # 3. The Audio Input Widget
    # NOTICE the 'key' argument. It changes every time we click reset.
    audio_value = st.audio_input(
        "Record your voice", key=f"audio_recorder_{st.session_state.voice_key}"
    )

    if audio_value:
        st.success("Audio captured!")

        # Step 1: Transcribe
        with st.spinner("Transcribing..."):
            user_text = transribe_audio(audio_value)
            st.write(f"**You said:** {user_text}")

        if user_text:
            # Step 2: Think
            with st.spinner("Thinking..."):
                response_text = get_ai_response(user_text)
                st.write(f"**AI:** {response_text}")

            # Step 3: Speak
            with st.spinner("Generating Voice..."):
                import nest_asyncio

                nest_asyncio.apply()
                asyncio.run(generate_audio(response_text, "response.mp3"))

                # Auto-play audio
                st.audio("response.mp3", format="audio/mp3", autoplay=True)

            # 4. THE FIX: A Button to Clear
            # When clicked, this runs 'reset_audio', increments the key, and reloads the page.
            st.button("🎤 Click to Record Again", on_click=reset_audio)


if __name__ == "__main__":
    main()
