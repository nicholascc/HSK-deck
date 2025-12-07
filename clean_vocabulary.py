#!/usr/bin/env python3
"""
Clean HSK vocabulary by using LLM to intelligently select forms.
Handles multi-pronunciation words and creates multiple cards when appropriate.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Paths (relative to script location)
SCRIPT_DIR = Path(__file__).parent.resolve()
HSK_JSON_DIR = SCRIPT_DIR / "complete-hsk-vocabulary/wordlists/exclusive/new"
CLEANED_DIR = SCRIPT_DIR / "data/cleaned"
CLEANED_DIR.mkdir(parents=True, exist_ok=True)

# DeepSeek client
client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)

SYSTEM_PROMPT = """You are a Chinese language expert. Analyze multi-pronunciation Chinese words and decide which form(s) to include as flashcards.

**Decision Rules:**
1. CREATE MULTIPLE CARDS when different pronunciations are both common (e.g., 了 le/liǎo, 还 hái/huán)
2. CREATE SINGLE CARD for the most common pronunciation, skip rare/surname/literary forms
3. For HSK 1-3: Prioritize spoken/practical usage
4. Prioritize: particles > conjunctions > common verbs/nouns > rare usage

**Output JSON array** with one object per flashcard:
[
  {
    "word": "了",
    "pinyin": "le",
    "meanings": ["aspect particle", "modal particle"],
    "hsk_level": 1,
    "classifier": "",
    "frequency": 3
  }
]

Keep meanings concise (max 5). Return ONLY the JSON array, no explanations."""


def extract_json_array(text: str) -> List[Dict]:
    """Extract JSON array from response with multiple fallback methods."""

    # Try 1: Direct parsing
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except:
        pass

    # Try 2: Extract from markdown code block
    match = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', text)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass

    # Try 3: Find JSON array pattern
    match = re.search(r'\[\s*\{[\s\S]*?\}\s*\]', text)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass

    print(f"  ✗ Could not extract JSON. Response preview:")
    print(f"    {text[:300]}...")
    return []


def process_chunk(words_chunk: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Send chunk to DeepSeek and get cleaned cards."""

    user_message = json.dumps(words_chunk, ensure_ascii=False, indent=2)

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3
            )

            result = response.choices[0].message.content.strip()
            cards = extract_json_array(result)

            if cards:
                print(f"  ✓ Received {len(cards)} cards")
                return cards
            else:
                print(f"  ⚠ Attempt {attempt + 1}/3 failed")
                if attempt < 2:
                    time.sleep(2)

        except Exception as e:
            print(f"  ✗ API error (attempt {attempt + 1}/3): {e}")
            if attempt < 2:
                time.sleep(2)

    print(f"  ✗ All attempts failed, skipping chunk")
    return []


def process_single_form_words(vocabulary: List[Dict[str, Any]], hsk_level: int) -> List[Dict[str, Any]]:
    """Convert single-form words directly to cards (no LLM needed)."""
    cards = []

    for entry in vocabulary:
        forms = entry.get('forms', [])
        if len(forms) == 1:
            form = forms[0]
            card = {
                "word": entry['simplified'],
                "pinyin": form['transcriptions']['pinyin'],
                "meanings": form['meanings'],
                "hsk_level": hsk_level,
                "classifier": ' / '.join(form.get('classifiers', [])),
                "frequency": entry.get('frequency', 999999)
            }
            cards.append(card)

    return cards


def clean_hsk_level(level: int, chunk_size: int = 20) -> List[Dict[str, Any]]:
    """Clean vocabulary for a single HSK level."""

    print(f"\n{'='*80}")
    print(f"CLEANING HSK LEVEL {level}")
    print(f"{'='*80}\n")

    # Load vocabulary
    json_file = HSK_JSON_DIR / f"{level}.json"
    print(f"Loading {json_file}...")
    with open(json_file, 'r', encoding='utf-8') as f:
        vocabulary = json.load(f)

    print(f"Total words: {len(vocabulary)}")

    # Separate single-form and multi-form words
    single_form = [w for w in vocabulary if len(w.get('forms', [])) == 1]
    multi_form = [w for w in vocabulary if len(w.get('forms', [])) > 1]

    print(f"Single-form: {len(single_form)} (direct conversion)")
    print(f"Multi-form: {len(multi_form)} (LLM analysis)")

    # Process single-form words
    all_cards = process_single_form_words(single_form, level)
    print(f"✓ Created {len(all_cards)} cards from single-form words")

    # Process multi-form words in chunks
    if multi_form:
        print(f"\nProcessing multi-form words in chunks of {chunk_size}...")

        # Prepare input for LLM
        llm_input = []
        for w in multi_form:
            llm_input.append({
                "simplified": w['simplified'],
                "hsk_level": level,
                "forms": w['forms'],
                "frequency": w.get('frequency', 999999)
            })

        # Process in chunks
        for i in range(0, len(llm_input), chunk_size):
            chunk = llm_input[i:i + chunk_size]
            chunk_num = (i // chunk_size) + 1
            total_chunks = (len(llm_input) + chunk_size - 1) // chunk_size

            print(f"\nChunk {chunk_num}/{total_chunks} ({len(chunk)} words):")
            cards = process_chunk(chunk)
            all_cards.extend(cards)
            time.sleep(1)  # Rate limiting

    print(f"\n{'='*80}")
    print(f"TOTAL: {len(all_cards)} cards for HSK {level}")
    print(f"{'='*80}")

    return all_cards


def save_cleaned_vocabulary(level: int, cards: List[Dict[str, Any]]):
    """Save cleaned vocabulary to JSON file."""
    output_file = CLEANED_DIR / f"{level}_cleaned.json"

    # Sort by frequency
    cards.sort(key=lambda x: x.get('frequency', 999999))

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cards, f, ensure_ascii=False, indent=2)

    print(f"✓ Saved to {output_file}")


def main():
    import sys

    print("="*80)
    print("HSK VOCABULARY CLEANER")
    print("="*80)
    print("\nUses DeepSeek to process multi-pronunciation words.")
    print("Creates multiple cards when appropriate (e.g., 了 le/liǎo)")
    print("="*80)

    # Check for command-line arguments (non-interactive mode)
    if len(sys.argv) > 1:
        # Non-interactive mode: python clean_vocabulary.py 1,2,3
        level_input = sys.argv[1]
        print(f"\nNon-interactive mode: processing levels {level_input}")
    else:
        # Interactive mode
        level_input = input("\nWhich HSK level(s)? [1-7 or 'all']: ").strip().lower()

    if level_input == 'all':
        levels = [1, 2, 3, 4, 5, 6, 7]
    else:
        try:
            levels = [int(x.strip()) for x in level_input.split(',')]
            if not all(1 <= level <= 7 for level in levels):
                print("❌ Invalid levels")
                return
        except ValueError:
            print("❌ Invalid input")
            return

    print(f"\nProcessing levels: {levels}")

    # Only prompt in interactive mode
    if len(sys.argv) <= 1:
        input("\nPress Enter to start...")

    # Process each level
    for level in levels:
        try:
            cards = clean_hsk_level(level)
            save_cleaned_vocabulary(level, cards)
        except Exception as e:
            print(f"\n❌ Error processing HSK {level}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("✅ COMPLETE")
    print("="*80)
    print(f"\nCleaned data saved to: {CLEANED_DIR}/")


if __name__ == "__main__":
    main()
