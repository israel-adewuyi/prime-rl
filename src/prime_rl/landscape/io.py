import csv
import hashlib
import json
from pathlib import Path

import verifiers as vf

from prime_rl.landscape.config import LandscapeConfig


def _format_message_content(content) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, indent=2)


def _format_messages(messages: list[dict]) -> str:
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = _format_message_content(msg.get("content", ""))
        lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines)


def _format_rollout_text(rollout: vf.State, alpha: float, beta: float) -> str:
    header = (
        f"=== alpha={alpha:.3f} beta={beta:.3f} | example_id={rollout.get('example_id')} | "
        f"task={rollout.get('task')} | reward={rollout.get('reward')} | "
        f"is_truncated={rollout.get('is_truncated')} | error={rollout.get('error')} ==="
    )
    blocks = [header]
    trajectory = rollout.get("trajectory") or []
    for idx, step in enumerate(trajectory, start=1):
        prompt = step.get("prompt") or []
        response = step.get("response")
        blocks.append(f"TURN {idx} PROMPT:")
        blocks.append(_format_messages(prompt))
        blocks.append("")
        blocks.append(f"TURN {idx} RESPONSE:")
        if response is None:
            blocks.append("")
        else:
            if hasattr(response, "model_dump"):
                response = response.model_dump()
            if isinstance(response, dict):
                choices = response.get("choices") or []
                if choices:
                    message = choices[0].get("message") or {}
                    blocks.append(_format_message_content(message.get("content", "")))
                else:
                    blocks.append(_format_message_content(response))
            else:
                blocks.append(_format_message_content(response))
        blocks.append("\n---\n")
    return "\n".join(blocks)


def write_metadata(config: LandscapeConfig, output_dir: Path) -> None:
    metadata_path = output_dir / config.sweep.metadata_file
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "output_dir": str(output_dir),
        "trainer": config.trainer.model_dump(mode="json"),
        "orchestrator": config.orchestrator.model_dump(mode="json"),
        "sweep": config.sweep.model_dump(mode="json"),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def append_result(output_path: Path, row: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()
    with open(output_path, "a", newline="") as f:
        fieldnames = list(row.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        else:
            with open(output_path, "r", newline="") as read_f:
                header = read_f.readline().strip()
            if header and header != ",".join(fieldnames):
                raise ValueError(
                    f"Existing results file has different header: {header}. "
                    f"Expected: {','.join(fieldnames)}. "
                    "Delete the file or change output_dir to continue."
                )
        writer.writerow(row)


def append_sampled_prompts(output_path: Path, examples: list[dict]) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prompts = [example.get("prompt") for example in examples]
    prompts_payload = json.dumps(prompts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    prompts_hash = hashlib.sha256(prompts_payload.encode("utf-8")).hexdigest()

    with open(output_path, "a", encoding="utf-8") as f:
        f.write(f"=== sampled_prompts count={len(examples)} sha256={prompts_hash} ===\n")
        for prompt in prompts:
            f.write(json.dumps(prompt, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            f.write("\n")
        f.write("\n")

    return prompts_hash


def append_rollouts(output_path: Path, rollouts: list[vf.State], alpha: float, beta: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        for rollout in rollouts:
            f.write(_format_rollout_text(rollout, alpha, beta))
            f.write("\n")
