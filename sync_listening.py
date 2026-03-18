#!/usr/bin/env python3
"""
Generate listening cards from YouTube transcripts and add them directly to Anki.
Uses your existing 'Chinese Audio -> Meaning' note model.
"""

import os
import re
import json
import time
import base64
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

# --- Configuration ---
DECK_NAME = "knowledge::mandarin listening::auto"
MODEL_NAME = "Chinese Audio -> Meaning"          # Your existing note model
ANKI_URL = "http://127.0.0.1:8765"
SCRIPT_DIR = Path(__file__).parent.resolve()
AUDIO_DIR = SCRIPT_DIR / "audio/listening"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# API clients
deepseek = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
eleven = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# ----------------------------------------------------------------------
# AnkiConnect helpers
# ----------------------------------------------------------------------
def anki(action, **params):
    try:
        resp = requests.post(ANKI_URL, json={"action": action, "params": params, "version": 6}).json()
        if resp.get("error"):
            print(f"Anki error: {resp['error']}")
        return resp.get("result")
    except Exception as e:
        print(f"Anki connection failed: {e}")
        return None

def get_known_hanzi():
    """Fetch all Hanzi fields from your listening deck."""
    note_ids = anki("findNotes", query=f'deck:"{DECK_NAME}"')
    if not note_ids:
        return []
    notes = anki("notesInfo", notes=note_ids)
    return [n["fields"]["Hanzi"]["value"] for n in notes if "Hanzi" in n["fields"]]

def upload_media(filename, data):
    """Upload a media file to Anki's collection. Data must be bytes."""
    try:
        # Encode binary data to base64 string
        encoded = base64.b64encode(data).decode('utf-8')
        result = anki("storeMediaFile", filename=filename, data=encoded)
        if result is None:
            print(f"❌ Failed to upload media: {filename}")
            return False
        return True
    except Exception as e:
        print(f"❌ Media upload error for {filename}: {e}")
        return False

# ----------------------------------------------------------------------
# Audio generation
# ----------------------------------------------------------------------
def generate_audio(text, idx, speed=0.85):
    """Generate audio with ElevenLabs at specified speed."""
    filename = f"listen_{int(time.time())}_{idx:03d}.mp3"
    path = AUDIO_DIR / filename
    try:
        # Add speed parameter to the generation
        audio = eleven.text_to_speech.convert(
            voice_id="pTOe8BQRdydOEIgv0wFL",      # Change to your preferred voice
            text=text,
            model_id="eleven_multilingual_v2",
            voice_settings={
                "speed": speed  # Set speed (0.85 = 15% slower)
            }
        )
        # Write to local file
        with open(path, "wb") as f:
            for chunk in audio:
                f.write(chunk)

        # Read the file bytes for Anki upload
        with open(path, "rb") as f:
            file_data = f.read()

        # Upload to Anki
        if not upload_media(filename, file_data):
            print(f"⚠ Audio file generated but not uploaded: {filename}")
            return None

        return filename
    except Exception as e:
        print(f"❌ Audio generation failed for '{text}': {e}")
        return None

# ----------------------------------------------------------------------
# AI prompt (in Mandarin)
# ----------------------------------------------------------------------
def get_ai_suggestions(transcript, known_list, num_cards=5):
    system_prompt = """你是一位专门研究「可理解性输入」教学法的中文听力专家。用户完全通过沉浸式听力学习中文，使用Anki进行间隔重复。

任务目标：
从提供的YouTube自动生成字幕中，提取最适合听力训练的自然语块。

核心原则：

    听觉优先：选择的短语必须听觉清晰、有辨识度，适合作为听力卡片

    自然语感：优先选择母语者日常会话中真正使用的表达

    难度适配：在用户已知内容（见已知短语列表）和视频难度之间找到“i+1”的黄金点

    避免重复：已知短语列表中已包含用户已掌握的短语。请勿选择与列表中任何短语完全相同的表达——精确重复对学习无用。但含义相近、结构不同的短语是完全可以的。

    初学者友好：用户是中文初学者，需要大量重复基础表达。请勿添加字幕中未出现的复杂内容——严格遵守视频实际使用的语言。

学习情境说明：

    用户已观看视频两遍，理解程度约达90%，并拥有完整的视频语境支持

    因此，所选短语不应比视频本身更难，且应避免选取视频中最难的表达

    学习流程：生成音频 → Anki卡片先播放音频，用户需理解其含义

短语筛选标准：

    长度：与视频中短语的自然长度保持一致，通常为2–6个汉字

    音频时长：控制在1–3秒，便于听力训练

    声调搭配：优先选择声调变化丰富的短语，便于训练听觉分辨

    边界清晰：短语前后有自然停顿感，避免吞音或连读过于严重的片段

特别注意事项：

    字幕可能因自动生成而有错别字、断句错误或漏字

    如果字幕片段明显错误，请根据上下文推断正确表达

    完全忽略无法理解或明显错误的字幕片段

输出格式：
必须返回严格的JSON，包含两个字段：
{
"phrases": [
{
"hanzi": "汉字短语",
"pinyin": "带声调的拼音",
"meaning": "英文释义",
"notes": "简短的教学提示（可选）"
}
],
"commentary": "三句英文注释，用|分隔。第一句：字幕质量问题及修正；第二句：基于已知短语观察到的用户水平；第三句：今日添加的短语类型说明"
}"""

    user_prompt = f"""请分析以下视频字幕，并完成上述任务。

已知短语列表（用户已经掌握的汉字）：
{json.dumps(known_list, ensure_ascii=False)}

字幕内容：
{transcript}

需要提取的短语数量：{num_cards}"""

    try:
        resp = deepseek.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        content = resp.choices[0].message.content
        # Sometimes DeepSeek wraps in markdown; try to extract JSON
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            return json.loads(match.group())
        else:
            print("No JSON found in AI response")
            return {"phrases": [], "commentary": ""}
    except Exception as e:
        print(f"AI request failed: {e}")
        return {"phrases": [], "commentary": ""}

# ----------------------------------------------------------------------
# Main workflow
# ----------------------------------------------------------------------
def run(video_url, num_cards=5, speed=0.85):
    print("=" * 60)
    print("YOUTUBE LISTENING CARD GENERATOR")
    print("=" * 60)
    print(f"Audio speed: {speed}x (1.0 = normal, <1 = slower, >1 = faster)")

    # 1. Extract video ID
    video_id = video_url.split("v=")[-1].split("&")[0]
    print(f"Video ID: {video_id}")

    # 2. Fetch transcript
    print("Fetching transcript...")
    try:
        ytt = YouTubeTranscriptApi()
        transcript_list = ytt.list(video_id)
        # Prefer Chinese
        transcript = transcript_list.find_transcript(['zh-Hans', 'zh-CN', 'zh']).fetch()
        full_text = " ".join([snippet.text for snippet in transcript])
        print(f"✓ Transcript length: {len(full_text)} chars")
    except Exception as e:
        print(f"❌ Transcript fetch failed: {e}")
        return

    # 3. Get known phrases from Anki
    known = get_known_hanzi()
    print(f"Found {len(known)} existing cards in deck")

    # 4. AI analysis
    print("Asking DeepSeek for micro-phrases...")
    result = get_ai_suggestions(full_text, known, num_cards)
    phrases = result.get("phrases", [])
    commentary = result.get("commentary", "")

    # Print commentary (canary)
    if commentary:
        print("\n" + "=" * 40)
        print("DEEPSEEK COMMENTARY")
        print("=" * 40)
        for i, line in enumerate(commentary.split("|"), 1):
            print(f"{i}. {line.strip()}")
        print("=" * 40 + "\n")

    if not phrases:
        print("No new phrases suggested. Exiting.")
        return

    # 5. Create deck if needed
    anki("createDeck", deck=DECK_NAME)

    # 6. Generate audio and add notes
    added = 0
    for idx, card in enumerate(phrases):
        # Generate audio with specified speed (this also uploads to Anki)
        audio_file = generate_audio(card["hanzi"], idx, speed=speed)
        if not audio_file:
            print(f"❌ Skipping card due to audio failure: {card['hanzi']}")
            continue

        # Prepare note
        note = {
            "deckName": DECK_NAME,
            "modelName": MODEL_NAME,
            "fields": {
                "Hanzi": card["hanzi"],
                "Pinyin": card["pinyin"],
                "Meaning": card["meaning"],
                "Audio": f"[sound:{audio_file}]",
                "Notes": card.get("notes", "")
            },
            "options": {"allowDuplicate": False}
        }

        # Add to Anki
        note_id = anki("addNote", note=note)
        if note_id:
            print(f"✅ Added: {card['hanzi']} — {card['meaning']}")
            added += 1
        else:
            print(f"⚠ Skipped (duplicate?): {card['hanzi']}")

        # Tiny delay to be nice to ElevenLabs
        time.sleep(0.3)

    print(f"\nDone. Added {added} new card(s).")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("-n", "--num-cards", type=int, default=5, help="Number of cards to generate")
    parser.add_argument("-s", "--speed", type=float, default=0.7, 
                       help="Audio speed (0.7-1.2)")
    args = parser.parse_args()
    run(args.url, args.num_cards, args.speed)