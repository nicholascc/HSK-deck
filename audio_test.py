import os
import time
from pathlib import Path
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

# 1. Setup paths (mimicking your script structure)
SCRIPT_DIR = Path(__file__).parent.resolve()
AUDIO_DIR = SCRIPT_DIR / "test_audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()

# 2. Initialize Client
api_key = os.getenv('ELEVENLABS_API_KEY')
if not api_key:
    print("❌ ERROR: ELEVENLABS_API_KEY not found in .env file.")
else:
    print(f"🔑 API Key found (starts with: {api_key[:5]}...)")

client = ElevenLabs(api_key=api_key)

def test_audio_gen(text_to_speak: str, filename: str):
    """Simplified version of your generation logic with verbose errors."""
    audio_path = AUDIO_DIR / filename
    
    print(f"🚀 Attempting to generate audio for: '{text_to_speak}'")
    
    try:
        # Using the same voice ID from your script
        audio_generator = client.text_to_speech.convert(
            voice_id="pTOe8BQRdydOEIgv0wFL", 
            text=text_to_speak,
            model_id="eleven_multilingual_v2"
        )

        with open(audio_path, 'wb') as f:
            for chunk in audio_generator:
                f.write(chunk)

        if audio_path.exists() and audio_path.stat().st_size > 0:
            print(f"✅ SUCCESS: File saved to {audio_path}")
            print(f"   File size: {audio_path.stat().st_size} bytes")
            return True
        else:
            print("❌ FAILURE: File was created but is empty.")
            return False

    except Exception as e:
        print(f"❌ API ERROR: {type(e).__name__}")
        print(f"   Details: {str(e)}")
        return False

if __name__ == "__main__":
    print("--- ElevenLabs Integration Test ---")
    
    # Test Case 1: Simple Chinese Sentence
    success = test_audio_gen("这是一次测试。", "test_sentence.mp3")
    
    if success:
        print("\n🎉 Audio generation is working correctly!")
    else:
        print("\n                     DEBUG CHECKLIST:")
        print("1. Is your API key valid and active?")
        print("2. Do you have remaining character credits on ElevenLabs?")
        print("3. Is the Voice ID 'pTOe8BQRdydOEIgv0wFL' still valid in your account?")
        print("4. Check if a firewall/VPN is blocking 'api.elevenlabs.io'.")
