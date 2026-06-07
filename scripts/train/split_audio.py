"""Extract per-sample user-turn audio from per-dialogue WAV files.

SpokenWOZ provides one WAV file per dialogue. Each turn in the JSON log has
word-level timing (BeginTime / EndTime in milliseconds). This script extracts
the user turn audio for each (system, user) sample pair.

Expected input structure:
  <audio-dir>/
    MUL0001.wav
    MUL0002.wav
    ...

Output structure:
  <output-dir>/
    MUL0001_1_2.wav   # sample k=0: sys=log[1], user=log[2]
    MUL0001_3_4.wav   # sample k=1: sys=log[3], user=log[4]
    ...

Usage:
  python scripts/train/split_audio.py \\
      --data     data/raw/train.json \\
      --audio-dir data/raw/audio \\
      --output-dir data/audio/train

  python scripts/train/split_audio.py \\
      --data     data/raw/test.json \\
      --audio-dir data/raw/audio \\
      --output-dir data/audio/test
"""

import argparse
import json
from pathlib import Path

import soundfile as sf


PADDING_MS = 100  # silence padding added before/after each user turn


def extract_samples(
    dialogue_id: str,
    log: list[dict],
    audio_path: Path,
    output_dir: Path,
) -> int:
    """Extract per-sample user-turn audio for one dialogue.

    Pairs: (log[2k+1]=system, log[2k+2]=user) for k = 0, 1, ...
    Output file: {dialogue_id}_{sys_idx}_{user_idx}.wav  (user audio only)

    Returns the number of samples successfully saved.
    """
    audio, sr = sf.read(str(audio_path), always_2d=False)
    total_samples = len(audio) if audio.ndim == 1 else audio.shape[0]

    saved = 0
    k = 0
    while True:
        sys_idx = 2 * k + 1
        user_idx = 2 * k + 2

        if user_idx >= len(log):
            break

        sys_turn = log[sys_idx]
        user_turn = log[user_idx]

        if sys_turn.get("tag") != "system" or user_turn.get("tag") != "user":
            k += 1
            continue

        words = user_turn.get("words", [])
        if not words:
            print(f"  [WARN] {dialogue_id} user turn {user_idx}: no word timing, skipping")
            k += 1
            continue

        begin_ms = max(0, words[0]["BeginTime"] - PADDING_MS)
        end_ms = words[-1]["EndTime"] + PADDING_MS

        begin_sample = int(begin_ms * sr / 1000)
        end_sample = min(total_samples, int(end_ms * sr / 1000))

        if begin_sample >= end_sample:
            print(f"  [WARN] {dialogue_id} user turn {user_idx}: empty segment, skipping")
            k += 1
            continue

        segment = audio[begin_sample:end_sample]
        out_path = output_dir / f"{dialogue_id}_{sys_idx}_{user_idx}.wav"
        sf.write(str(out_path), segment, sr)
        saved += 1
        k += 1

    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="SpokenWOZ JSON file")
    parser.add_argument("--audio-dir", required=True, help="Directory with per-dialogue WAV files")
    parser.add_argument("--output-dir", required=True, help="Output directory for per-sample WAV files")
    parser.add_argument("--ext", default="wav", help="Audio file extension (default: wav)")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.data, encoding="utf-8") as f:
        data: dict = json.load(f)

    total_saved = 0
    skipped_dialogues = 0

    for dialogue_id, dialogue in data.items():
        log = dialogue.get("log", [])
        audio_path = audio_dir / f"{dialogue_id}.{args.ext}"

        if not audio_path.exists():
            print(f"[WARN] Audio not found: {audio_path}, skipping dialogue")
            skipped_dialogues += 1
            continue

        saved = extract_samples(dialogue_id, log, audio_path, output_dir)
        total_saved += saved

    print(f"\nDone. {total_saved} sample files written to {output_dir}/")
    if skipped_dialogues:
        print(f"Skipped {skipped_dialogues} dialogues (audio file not found)")


if __name__ == "__main__":
    main()
