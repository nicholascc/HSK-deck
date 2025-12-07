# HSK Anki Deck Generator

Generate Anki flashcard decks for HSK Chinese vocabulary (levels 1-7) with AI-generated example sentences and native audio.

## Overview

This project creates Anki decks for studying Chinese using the new HSK 3.0 standard vocabulary lists. Supports any combination of HSK levels 1-7. Each card includes:
- Target vocabulary word with pinyin
- AI-generated example sentence using constrained vocabulary
- Native Chinese audio (I used ElevenLabs TTS because I had free credits. Feel free to pick whichever provider you prefer)
- English translations
- Progressive difficulty (sentences only use previously learned words)

## The Approach

### Constrained Sentence Generation

The key insight is that example sentences should only use vocabulary the learner already knows. This prevents the frustrating experience of looking up every word in the example sentence.

**How it works:**
1. Cards are sorted by frequency (most common words first)
2. Each sentence only uses:
   - Words learned before this card
   - A seed vocabulary of ~80 common function words (的, 了, 是, etc.)
   - The target word itself
3. DeepSeek AI generates contextually appropriate sentences within these constraints. Deepseek is preferred because presumably it has a higher percentage of chinese vocabulary in its training data, and caching makes generating new cards with previously learned vocabulary very cheap.

**Example progression:**
- Card 1 learns "你" (you) → sentence uses only seed words + 你
- Card 50 learns "喜欢" (like) → sentence can use 你 + 49 other learned words + 喜欢
- Card 500 learns "复杂" (complex) → sentence can use all 499 previous words + 复杂

### Two-Stage Cleaning

**Stage 1: Vocabulary Cleaning** (`clean_vocabulary.py`)
- Handles multi-pronunciation words (e.g., 了 can be "le" or "liǎo")
- Creates separate cards for each distinct pronunciation
- Filters out rare/literary/surname usages
- Uses DeepSeek to make intelligent decisions about which readings to include, since common words can have distinct readings that are equally common.

**Stage 2: Deck Generation** (`generate_deck.py`)
- Generates contextually appropriate sentences with vocabulary constraints
- Generates audio for each sentence using ElevenLabs TTS
- Builds Anki package with subdecks for each HSK level

## Requirements

```bash
# Python packages
uv pip install openai elevenlabs genanki python-dotenv

# API Keys (in .env file)
DEEPSEEK_API_KEY=your_key_here
ELEVENLABS_API_KEY=your_key_here
```

## Usage

### 1. Clean Vocabulary (First Time Only)

```bash
uv run clean_vocabulary.py
# Select HSK levels when prompted
# Examples: "1", "1,2", "1,2,3", or "1,2,3,4,5,6,7"
```

This processes the raw HSK vocabulary and creates cleaned JSON files in `data/cleaned/`.

### 2. Generate Deck

```bash
uv run generate_deck.py
# Select HSK levels when prompted
# Examples:
#   "1"       → HSK_1_cleaned.apkg
#   "1,2,3"   → HSK_1-3_cleaned.apkg
#   "4,5"     → HSK_4-5_cleaned.apkg
#   "1,2,3,4,5,6,7" → HSK_1-7_cleaned.apkg
```

This generates:
- AI sentences for each card
- Native audio files
- Complete Anki deck (.apkg file)

Output: `decks/HSK_{levels}_cleaned.apkg`

## Architecture

```
HSK-deck/
├── complete-hsk-vocabulary/    # Raw HSK vocabulary lists (source data)
├── data/
│   └── cleaned/                # Cleaned vocabulary (after Stage 1)
├── audio/
│   └── sentences/              # Generated audio files
├── decks/                      # Final Anki decks (.apkg)
├── clean_vocabulary.py         # Stage 1: Vocabulary cleaning
└── generate_deck.py            # Stage 2: Deck generation
```

## Features

- **Parallel Processing**: Batch sentence generation and concurrent audio generation
- **Rate Limit Handling**: Automatic retry with exponential backoff for API limits
- **Smart Vocabulary**: Only uses words the learner should already know
- **Native Audio**: High-quality Chinese TTS from ElevenLabs
- **Multiple Pronunciations**: Separate cards for words like 了 (le/liǎo), 行 (xíng/háng)

## Performance

Parallel processing with 10 sentence workers and 5 audio workers:

- **HSK 1** (~530 cards): ~10-15 minutes
- **HSK 1-3** (~2,300 cards): ~30-45 minutes
- **HSK 1-7** (~11,500 cards): ~2-3 hours

## Card Format

**Front:**
- Target word in large text
- Example sentence (target word highlighted in green)

**Back:**
- Pinyin for word and sentence
- English translation (hidden by default, click to reveal)
- Word meaning (hidden by default, click to reveal)
- Audio plays automatically

## Troubleshooting

### Most Audio Files Failed (0 bytes)
- This happens when hitting API concurrency limits
- The script uses 5 workers to avoid this in Elevenlabs, but other TTS providers may have different quotas.
- Automatic retry logic handles transient failures

### API Quota Exceeded
- **DeepSeek**: Check usage at https://platform.deepseek.com/usage
- **ElevenLabs**: Check quota at https://elevenlabs.io/app

### Import Into Anki
1. Open Anki
2. File → Import
3. Select `decks/HSK_X-Y_cleaned.apkg`
4. Done!

## Credits

- **Vocabulary Data**: [complete-hsk-vocabulary](https://github.com/drkameleon/complete-hsk-vocabulary)
- **Sentence Generation System Prompt**: [@0bNARA](https://x.com/0bNARA) (with minimal modifications)

## License

MIT
