"""Convert SpokenWOZ data to GRPO-format JSONL for incremental DST training.

Expected SpokenWOZ raw format (e.g. train.json):
  {
    "DIALOGUE_ID": {
      "log": [
        {
          "tag":  "user",
          "text": "yes , i'm looking for a restaurant .",
          "metadata": {},
          "words": [{"Word": "yes", "BeginTime": 6550, "EndTime": 6857, ...}, ...]
        },
        {
          "tag":  "system",
          "text": "okay , any requirement .",
          "metadata": {
            "restaurant": {"book": {"booked": []}, "semi": {"area": "west", ...}},
            ...
          },
          "words": [...]
        },
        ...
      ]
    }
  }

Turn ordering: user-first (log[0].tag == "user").
Belief states (MultiWOZ 2.1 nested format) are stored on system turns.

Each output sample corresponds to one (system, user) pair:
  - system turn:  log[2k+1]   (k = 0, 1, 2, ...)
  - user turn:    log[2k+2]
  - prev_state:   flatten(log[2k+1].metadata)   — state before this exchange
  - curr_state:   flatten(log[2k+3].metadata)   — state after user's reply

Input to the model:
  - Text: dialogue history including the current system turn (log[0..2k+1])
  - Audio: current user turn only

Audio files must be pre-extracted with split_audio.py. Each sample references:
  audios: ["{dialogue_id}_{sys_idx}_{user_idx}.wav"]  (named with both indices for identification)

At inference, set --audio-base-dir to the audio output directory.

Usage:
  # 1. Split audio first (see split_audio.py)
  # 2. Then prepare JSONL:
  python scripts/train/prepare_data.py \\
      --data      data/raw/train.json \\
      --output    data/train.jsonl

  python scripts/train/prepare_data.py \\
      --data      data/raw/test.json \\
      --output    data/test.jsonl
"""

import argparse
import json
from pathlib import Path


SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "incremental.txt"


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


def flatten_belief_state(metadata: dict) -> dict:
    """Flatten MultiWOZ 2.1 nested metadata to {domain: {slot: value}}.

    Input:  {"hotel": {"semi": {"pricerange": "cheap", "area": ""}, "book": {...}}}
    Output: {"hotel": {"pricerange": "cheap"}}

    Filters out empty strings, "not mentioned", and the "booked" list field.
    """
    flat: dict = {}
    for domain, domain_data in metadata.items():
        if not isinstance(domain_data, dict):
            continue
        slots: dict = {}
        if "semi" in domain_data or "book" in domain_data:
            for section in ("semi", "book"):
                for slot, value in domain_data.get(section, {}).items():
                    if slot == "booked" or not value or value == "not mentioned":
                        continue
                    slots[slot] = value
        else:
            for slot, value in domain_data.items():
                if not value or value == "not mentioned":
                    continue
                slots[slot] = value
        if slots:
            flat[domain] = slots
    return flat


def compute_diff_ops(prev: dict, curr: dict) -> list[str]:
    """Return ordered list of diff operations (set/update/delete) from prev to curr."""
    ops: list[str] = []
    for domain in sorted(set(list(prev.keys()) + list(curr.keys()))):
        prev_slots = prev.get(domain, {})
        curr_slots = curr.get(domain, {})
        for slot in sorted(curr_slots.keys()):
            value = curr_slots[slot]
            if slot not in prev_slots:
                ops.append(f"set({domain}.{slot}={value})")
            elif prev_slots[slot] != value:
                ops.append(f"update({domain}.{slot}={value})")
        for slot in sorted(prev_slots.keys()):
            if slot not in curr_slots:
                ops.append(f"delete({domain}.{slot})")
    return ops


def build_user_message(history_lines: list[str], prev_state: dict) -> str:
    """Build GRPO-format user message.

    Format:
        [Dialogue History]        (omitted when empty)
        User: ...
        System: ...
        ...
        System: ...   ← includes current system turn

        [Previous State]
        {"domain": {"slot": "value"}}

        [New Audio]
        <audio>
    """
    parts: list[str] = []
    if history_lines:
        parts.append("[Dialogue History]")
        parts.extend(history_lines)
        parts.append("")
    parts.append("[Previous State]")
    parts.append(json.dumps(prev_state, ensure_ascii=False))
    parts.append("")
    parts.append("[New Audio]")
    parts.append("<audio>")
    return "\n".join(parts)


def build_solution(sys_text: str, user_text: str, prev: dict, curr: dict) -> str:
    """Build gold solution string."""
    ops = compute_diff_ops(prev, curr)
    answer = "\n".join(ops)
    transcript = f"System: {sys_text}\nUser: {user_text}"
    return f"<transcript>\n{transcript}\n</transcript>\n<answer>{answer}</answer>"


def process_dialogue(
    dialogue_id: str,
    log: list[dict],
    system_prompt: str,
) -> list[dict]:
    """Convert one dialogue into GRPO samples.

    Pairs: (log[2k+1]=system, log[2k+2]=user) for k = 0, 1, ...
    log[0] (first user turn) is always included as the first history line.
    """
    if len(log) < 3:
        return []

    samples: list[dict] = []

    # log[0] is always a user turn; it goes directly into history
    history_lines: list[str] = [f"User: {log[0]['text'].strip()}"]

    k = 0
    while True:
        sys_idx = 2 * k + 1
        user_idx = 2 * k + 2
        next_sys_idx = 2 * k + 3

        if user_idx >= len(log):
            break  # no more complete (system, user) pairs

        sys_turn = log[sys_idx]
        user_turn = log[user_idx]

        if sys_turn.get("tag") != "system" or user_turn.get("tag") != "user":
            # Unexpected ordering; skip this pair
            k += 1
            continue

        sys_text = sys_turn["text"].strip()
        user_text = user_turn["text"].strip()

        prev_state = flatten_belief_state(sys_turn.get("metadata", {}))
        if next_sys_idx < len(log):
            curr_state = flatten_belief_state(log[next_sys_idx].get("metadata", {}))
        else:
            curr_state = prev_state  # last pair: no further annotation

        # History includes current system turn (text); audio is user turn only
        history_with_sys = list(history_lines) + [f"System: {sys_text}"]

        samples.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": build_user_message(history_with_sys, prev_state)},
            ],
            "audios": [f"{dialogue_id}_{sys_idx}_{user_idx}.wav"],
            "solution": build_solution(sys_text, user_text, prev_state, curr_state),
            "belief_state": json.dumps(curr_state, ensure_ascii=False),
            "prev_belief_state": json.dumps(prev_state, ensure_ascii=False),
            "dialogue_id": dialogue_id,
            "turn_idx": k,
        })

        history_lines.append(f"System: {sys_text}")
        history_lines.append(f"User: {user_text}")
        k += 1

    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="SpokenWOZ JSON file")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--system-prompt", default=None, help="Override system prompt file")
    args = parser.parse_args()

    system_prompt = load_system_prompt()
    if args.system_prompt:
        system_prompt = Path(args.system_prompt).read_text(encoding="utf-8").strip()

    with open(args.data, encoding="utf-8") as f:
        data: dict = json.load(f)

    samples: list[dict] = []
    skipped = 0
    for dialogue_id, dialogue in data.items():
        log = dialogue.get("log", [])
        result = process_dialogue(dialogue_id, log, system_prompt)
        if not result:
            skipped += 1
        samples.extend(result)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Wrote {len(samples)} samples from {len(data) - skipped} dialogues → {args.output}")
    if skipped:
        print(f"Skipped {skipped} dialogues (too short)")


if __name__ == "__main__":
    main()
