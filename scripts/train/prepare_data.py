"""Convert SpokenWOZ data to GRPO-format JSONL for incremental DST training.

Expected SpokenWOZ raw format (e.g. train_v1.0.json):
  {
    "DIALOGUE_ID": {
      "log": [
        {"text": "...", "metadata": {}, "wav": "DIALOGUE_ID/0.wav"},
        {"text": "...", "metadata": {}, "wav": "DIALOGUE_ID/1.wav"},
        ...
      ]
    }
  }

Turn ordering: system-first (log[0] = SYSTEM, log[1] = USER, log[2] = SYSTEM, ...).
Belief states follow MultiWOZ 2.1 annotation and are stored on SYSTEM turns (even indices).
Audio paths in "wav" fields are relative to --audio-dir.

Each output record contains:
  - messages:          [system_prompt, user_message]
  - audios:            [system_wav_path, user_wav_path]  (relative paths)
  - solution:          "<transcript>...</transcript><answer>diff_ops</answer>"
  - belief_state:      JSON string of current state (after this turn)
  - prev_belief_state: JSON string of previous state (before this turn)
  - dialogue_id:       string identifier
  - turn_idx:          0-based index within dialogue

Usage:
  python scripts/train/prepare_data.py \\
      --data data/raw/train_v1.0.json \\
      --output data/train.jsonl

  python scripts/train/prepare_data.py \\
      --data data/raw/test_v1.0.json \\
      --output data/test.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Optional


SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "incremental.txt"


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


def flatten_belief_state(metadata: dict) -> dict:
    """Flatten MultiWOZ metadata to {domain: {slot: value}}.

    Handles both:
    - Already-flat:    {"hotel": {"pricerange": "cheap"}}
    - MultiWOZ nested: {"hotel": {"semi": {"pricerange": "cheap"}, "book": {...}}}

    Filters out empty strings and the "booked" list field.
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
    """Return diff operations (set/update/delete) from prev to curr state."""
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


def build_user_message(history_lines: list[str], prev_state: dict, n_audio: int) -> str:
    """Build user message with dialogue history, previous state, and audio placeholders.

    Format:
        [Dialogue History]
        System: ...
        User: ...

        [Previous State]
        {...}

        [New Audio]
        <audio>
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
    for _ in range(n_audio):
        parts.append("<audio>")
    return "\n".join(parts)


def build_solution(sys_text: str, user_text: str, prev: dict, curr: dict) -> str:
    """Build gold solution string."""
    transcript = f"System: {sys_text}\nUser: {user_text}"
    ops = compute_diff_ops(prev, curr)
    answer = "\n".join(ops)
    return f"<transcript>\n{transcript}\n</transcript>\n<answer>{answer}</answer>"


def process_dialogue(
    dialogue_id: str,
    log: list[dict],
    system_prompt: str,
) -> list[dict]:
    """Convert one dialogue's log into GRPO samples.

    Iterates over (system, user) turn pairs: (log[2k], log[2k+1]).
    Belief state before each pair = log[2k].metadata.
    Belief state after each pair  = log[2k+2].metadata (next system turn).
    """
    samples: list[dict] = []
    history_lines: list[str] = []

    k = 0
    while 2 * k + 1 < len(log):
        sys_turn = log[2 * k]
        user_turn = log[2 * k + 1]

        sys_text = sys_turn.get("text", "").strip()
        user_text = user_turn.get("text", "").strip()
        sys_wav: Optional[str] = sys_turn.get("wav") or None
        user_wav: Optional[str] = user_turn.get("wav") or None

        prev_state = flatten_belief_state(sys_turn.get("metadata", {}))
        if 2 * k + 2 < len(log):
            curr_state = flatten_belief_state(log[2 * k + 2].get("metadata", {}))
        else:
            curr_state = prev_state

        audios = [w for w in [sys_wav, user_wav] if w]
        user_msg = build_user_message(list(history_lines), prev_state, n_audio=len(audios))
        solution = build_solution(sys_text, user_text, prev_state, curr_state)

        samples.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_msg},
            ],
            "audios":            audios,
            "solution":          solution,
            "belief_state":      json.dumps(curr_state,  ensure_ascii=False),
            "prev_belief_state": json.dumps(prev_state,  ensure_ascii=False),
            "dialogue_id":       dialogue_id,
            "turn_idx":          k,
        })

        history_lines.append(f"System: {sys_text}")
        history_lines.append(f"User: {user_text}")
        k += 1

    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",          required=True, help="SpokenWOZ JSON file")
    parser.add_argument("--output",        required=True, help="Output JSONL path")
    parser.add_argument("--system-prompt", default=None,  help="Override system prompt file")
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
        if len(log) < 2:
            skipped += 1
            continue
        samples.extend(process_dialogue(dialogue_id, log, system_prompt))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Wrote {len(samples)} samples from {len(data) - skipped} dialogues → {args.output}")
    if skipped:
        print(f"Skipped {skipped} dialogues with fewer than 2 turns")


if __name__ == "__main__":
    main()
