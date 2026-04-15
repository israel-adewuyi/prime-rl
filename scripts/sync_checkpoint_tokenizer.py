from argparse import ArgumentParser
from pathlib import Path

from transformers import AutoTokenizer


def parse_args():
    parser = ArgumentParser(description="Sync tokenizer files into RL checkpoint directories.")
    parser.add_argument(
        "--tokenizer-source",
        required=True,
        help="Base model or local tokenizer path to load from, e.g. Qwen/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument(
        "--target-dir",
        action="append",
        default=[],
        help="Checkpoint weights directory to update. Can be passed multiple times.",
    )
    parser.add_argument(
        "--checkpoints-root",
        help="Root directory containing grpo_steps/step_*/weights directories.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the tokenizer.",
    )
    parser.add_argument(
        "--fix-mistral-regex",
        action="store_true",
        help="Pass fix_mistral_regex=True when loading the tokenizer.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the directories that would be updated without writing files.",
    )
    return parser.parse_args()


def resolve_targets(target_dirs: list[str], checkpoints_root: str | None) -> list[Path]:
    targets = [Path(target_dir).resolve() for target_dir in target_dirs]

    if checkpoints_root is not None:
        root = Path(checkpoints_root).resolve()
        targets.extend(sorted(root.glob("step_*/weights")))

    unique_targets = []
    seen = set()
    for target in targets:
        if target in seen:
            continue
        seen.add(target)
        unique_targets.append(target)

    if not unique_targets:
        raise ValueError("Provide at least one --target-dir or a --checkpoints-root")

    for target in unique_targets:
        if not target.exists():
            raise ValueError(f"Target directory does not exist: {target}")
        if not target.is_dir():
            raise ValueError(f"Target path is not a directory: {target}")

    return unique_targets


def main() -> None:
    args = parse_args()
    targets = resolve_targets(args.target_dir, args.checkpoints_root)

    print(f"Loading tokenizer from: {args.tokenizer_source}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_source,
        trust_remote_code=args.trust_remote_code,
        fix_mistral_regex=args.fix_mistral_regex,
    )

    for target in targets:
        print(f"{'Would update' if args.dry_run else 'Updating'}: {target}")
        if args.dry_run:
            continue
        tokenizer.save_pretrained(target)

    print(f"Processed {len(targets)} checkpoint director{'y' if len(targets) == 1 else 'ies'}.")


if __name__ == "__main__":
    main()
