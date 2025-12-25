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
def main():
    st.title("🎙️ Free Real-Time Voice Agent")

    # New Streamlit Audio Input Widget
    audio_value = st.audio_input("Record your voice")

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

            # Step 3: Speak (Using Free Edge-TTS)
            with st.spinner("Generating Voice..."):
                # Run async function in sync Streamlit
                import nest_asyncio

                nest_asyncio.apply()
                asyncio.run(generate_audio(response_text, "response.mp3"))

                # Play the result in the browser
                st.audio("response.mp3", format="audio/mp3", autoplay=True)


if __name__ == "__main__":
    main()
