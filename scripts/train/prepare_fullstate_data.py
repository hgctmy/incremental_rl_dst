"""Convert incremental DST data to full-state DST format.

Transforms incremental DST data (diff operations output) into full-state DST
data (complete belief state JSON output). Handles both GRPO and SFT multimodal
input formats.

Key transformations:
  - Remove [Previous State] section from user input
  - Replace system prompt with full-state version
  - Change output from diff ops to full belief state JSON
  - Keep full dialogue history (same as incremental)

Usage:
  # Convert SFT multimodal test data:
  python scripts/train/prepare_fullstate_data.py \
      --input data/test.jsonl \
      --output data/fullstate_test.jsonl \
      --format sft

  # Convert GRPO training data:
  python scripts/train/prepare_fullstate_data.py \
      --input data/train.jsonl \
      --output data/fullstate_train.jsonl \
      --format grpo
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional


SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "fullstate.txt"


def load_system_prompt() -> str:
    """Load the full-state system prompt."""
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


def _extract_history_lines(text: str) -> list[str]:
    """Extract dialogue history lines from user message text.

    Parses text like:
        [Dialogue History]
        User: hello.
        System: hi.
        User: looking for restaurant.

        [Previous State]
        ...

    Returns list of utterance lines (e.g., ["User: hello.", "System: hi.", ...]).
    """
    # Find the [Dialogue History] section
    m = re.search(r'\[Dialogue History\]\s*\n(.*?)(?:\n\s*\n|\[Previous State\]|\[New Audio\])', text, re.DOTALL)
    if not m:
        return []

    lines = []
    for line in m.group(1).strip().split('\n'):
        line = line.strip()
        if line:
            lines.append(line)
    return lines


def _extract_transcript_text(text: str) -> Optional[str]:
    """Extract content between <transcript>...</transcript> tags."""
    m = re.search(r'<transcript>(.*?)</transcript>', text, re.DOTALL)
    return m.group(1).strip() if m else None


def _transcript_to_utterance_lines(transcript: str) -> list[str]:
    """Split transcript text into individual utterance lines.

    E.g., "System: hello.\\nUser: hi." -> ["System: hello.", "User: hi."]
    """
    lines = []
    for line in transcript.strip().split('\n'):
        line = line.strip()
        if line:
            lines.append(line)
    return lines


def _build_fullstate_user_text(history_lines: list[str]) -> str:
    """Build user message text for full-state format.

    Format:
        [Dialogue History]
        System: ...
        User: ...

        [New Audio]

    Or if no history:
        [New Audio]
    """
    parts = []
    if history_lines:
        parts.append("[Dialogue History]")
        parts.extend(history_lines)
        parts.append("")  # blank line

    parts.append("[New Audio]")
    return "\n".join(parts)


def _build_fullstate_solution(transcript: str, belief_state: Any) -> str:
    """Build the solution/output text for full-state format.

    Format:
        <transcript>
        System: ...
        User: ...
        </transcript>
        <answer>{"domain":{"slot":"value"}}</answer>
    """
    if isinstance(belief_state, str):
        state_str = belief_state
    else:
        state_str = json.dumps(belief_state, ensure_ascii=False)
    return f"<transcript>\n{transcript}\n</transcript>\n<answer>{state_str}</answer>"


def _get_turn_idx(sample: dict[str, Any]) -> int:
    """Extract turn index for ordering within a dialogue."""
    turn_idx = sample.get("turn_idx", -1)
    if turn_idx < 0 and "id" in sample:
        parts = sample["id"].rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            turn_idx = int(parts[1])
    return turn_idx


def _get_dialogue_id(sample: dict[str, Any]) -> str:
    """Extract dialogue ID from a sample."""
    if "dialogue_id" in sample:
        return sample["dialogue_id"]
    if "id" in sample:
        parts = sample["id"].rsplit("_", 1)
        if len(parts) == 2:
            return parts[0]
    return "unknown"


def _is_multimodal_content(content: Any) -> bool:
    """Check if content is in multimodal list format (vs plain string)."""
    return isinstance(content, list)


def _extract_user_text(sample: dict[str, Any]) -> str:
    """Extract user message text from a sample (handles both formats)."""
    for msg in sample["messages"]:
        if msg["role"] == "user":
            content = msg["content"]
            if _is_multimodal_content(content):
                return "".join(p["text"] for p in content if p.get("type") == "text")
            return content
    return ""


def _extract_audio_info(sample: dict[str, Any]) -> tuple[list[str], list[dict]]:
    """Extract audio paths and multimodal audio parts from a sample.

    Returns:
        (audio_paths, audio_parts):
            audio_paths: list of audio file paths (for GRPO format)
            audio_parts: list of audio content parts (for SFT multimodal format)
    """
    audio_paths = sample.get("audios", [])
    audio_parts = []

    for msg in sample["messages"]:
        if msg["role"] == "user" and _is_multimodal_content(msg["content"]):
            for part in msg["content"]:
                if part.get("type") == "audio":
                    audio_parts.append(part)

    return audio_paths, audio_parts


def _extract_solution_text(sample: dict[str, Any]) -> str:
    """Extract the gold solution/assistant text from a sample."""
    if "solution" in sample:
        return sample["solution"]

    for msg in sample["messages"]:
        if msg["role"] == "assistant":
            content = msg["content"]
            if _is_multimodal_content(content):
                return "".join(p["text"] for p in content if p.get("type") == "text")
            return content
    return ""


def _get_belief_state(sample: dict[str, Any]) -> Any:
    """Get the belief state from a sample (dict or string)."""
    bs = sample.get("belief_state", {})
    if isinstance(bs, str):
        try:
            return json.loads(bs) if bs else {}
        except json.JSONDecodeError:
            return {}
    return bs


def _remove_previous_state(text: str) -> str:
    """Remove the [Previous State] section from user message text.

    Keeps [Dialogue History] and [New Audio] sections intact.
    """
    # Remove [Previous State] ... up to [New Audio] or end, keeping [New Audio]
    text = re.sub(
        r'\[Previous State\].*?(?=\[New Audio\]|\Z)',
        '',
        text,
        flags=re.DOTALL,
    )
    # Clean up extra blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _convert_single_sample(
    sample: dict[str, Any],
    system_prompt: str,
    output_format: str,
) -> dict[str, Any]:
    """Convert a single incremental sample to full-state format.

    Each sample already contains the full dialogue history in its user message.
    We just need to:
    - Remove [Previous State] section
    - Replace system prompt
    - Change output to full belief state JSON
    """
    # Extract existing user text and remove [Previous State]
    user_text_raw = _extract_user_text(sample)
    user_text = _remove_previous_state(user_text_raw)

    # Get audio info
    audio_paths, audio_parts = _extract_audio_info(sample)

    # Get transcript from solution
    solution_text = _extract_solution_text(sample)
    transcript = _extract_transcript_text(solution_text)
    transcript_str = transcript if transcript else ""

    # Get belief state for full-state output
    belief_state = _get_belief_state(sample)

    # Build solution
    fullstate_solution = _build_fullstate_solution(transcript_str, belief_state)

    if output_format == "grpo":
        return _build_grpo_sample(
            sample, system_prompt, user_text, audio_paths,
            fullstate_solution, belief_state,
        )
    else:  # sft
        return _build_sft_sample(
            sample, system_prompt, user_text, audio_parts,
            fullstate_solution, belief_state, transcript_str,
        )


def convert_samples(
    samples: list[dict[str, Any]],
    system_prompt: str,
    output_format: str,
) -> list[dict[str, Any]]:
    """Convert incremental DST samples to full-state format.

    Each sample is converted independently:
    - Keeps full dialogue history (already in the user message)
    - Removes [Previous State]
    - Changes output to full belief state JSON
    """
    converted = []
    total = len(samples)

    for i, sample in enumerate(samples):
        converted.append(_convert_single_sample(sample, system_prompt, output_format))
        if (i + 1) % 10000 == 0:
            print(f"[INFO] Converted {i + 1}/{total} samples...", flush=True)

    return converted


def _build_grpo_sample(
    original: dict[str, Any],
    system_prompt: str,
    user_text: str,
    audio_paths: list[str],
    solution: str,
    belief_state: Any,
) -> dict[str, Any]:
    """Build a GRPO-format sample."""
    # Add <audio> placeholder only if not already present in the text
    if "<audio>" in user_text:
        user_content = user_text
    else:
        user_content = user_text + "\n<audio>"

    sample: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "solution": solution,
        "belief_state": json.dumps(belief_state, ensure_ascii=False) if isinstance(belief_state, dict) else belief_state,
    }

    if audio_paths:
        sample["audios"] = audio_paths

    return sample


def _build_sft_sample(
    original: dict[str, Any],
    system_prompt: str,
    user_text: str,
    audio_parts: list[dict],
    solution: str,
    belief_state: Any,
    transcript: str,
) -> dict[str, Any]:
    """Build an SFT multimodal format sample."""
    user_content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    user_content.extend(audio_parts)

    sample: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": solution}]},
        ],
        "belief_state": belief_state,
        "transcript": transcript,
    }

    # Preserve metadata from original
    if "id" in original:
        sample["id"] = original["id"]
    if "dialogue_id" in original:
        sample["dialogue_id"] = original["dialogue_id"]
    if "turn_idx" in original:
        sample["turn_idx"] = original["turn_idx"]

    return sample


def main():
    parser = argparse.ArgumentParser(
        description="Convert incremental DST data to full-state DST format"
    )
    parser.add_argument("--input", required=True, help="Input JSONL file (incremental format)")
    parser.add_argument("--output", required=True, help="Output JSONL file (full-state format)")
    parser.add_argument(
        "--format",
        choices=["grpo", "sft"],
        default="sft",
        help="Output format: grpo (for training) or sft (for inference/test)",
    )
    args = parser.parse_args()

    # Load system prompt
    system_prompt = load_system_prompt()

    # Load input data
    print(f"[INFO] Loading data from {args.input}")
    samples = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"[INFO] Loaded {len(samples)} samples")

    # Convert
    converted = convert_samples(samples, system_prompt, args.format)

    # Write output
    with open(args.output, "w", encoding="utf-8") as f:
        for sample in converted:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"[INFO] Converted {len(converted)} samples -> {args.output}")
    print(f"[INFO] Format: {args.format}")


if __name__ == "__main__":
    main()
