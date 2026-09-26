from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Published-data motor case: prepare, train, audit and export (no full training by default).")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Generate provenance-labelled input CSVs; idempotent.")
    prep.add_argument("--fetch", action="store_true", help="Download the unchanged open-access source PDF (network required).")
    commands.add_parser("validate", help="Audit source values, compatibility and integer units.")
    for command in ["train", "baselines", "evaluate", "aggregate", "export-paper", "run"]:
        sub = commands.add_parser(command)
        sub.add_argument("--profile", choices=["smoke", "full"], default="smoke")
        sub.add_argument("--output", help="Separate result root; profile subdirectory is added.")
        if command in ["train", "baselines", "evaluate", "run"]:
            sub.add_argument("--resume", action="store_true", help="Continue matching checkpoints / skip completed matching results.")
        if command in ["train", "evaluate", "run"]:
            sub.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="PPO/evaluation device; rollout workers remain on CPU.")
        if command == "train":
            sub.add_argument("--stop-after", type=int, help="Stop at this absolute iteration for interruption/resume checks; not a smaller formal budget.")
            sub.add_argument("--methods", nargs="+", choices=["HCMAGRL", "Flat", "MLP"])
        if command == "export-paper":
            sub.add_argument("--paper", help="Paper repository (otherwise HCMAGRL_PAPER or sibling checkout).")
            sub.add_argument("--draft", action="store_true", help="Export source table, schematic and explicitly pending result placeholders; no smoke numbers.")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            from .data import prepare
            prepare(args.fetch)
        elif args.command == "validate":
            from .data import validate_data
            validate_data()
        elif args.command == "train":
            from .learning import train
            train(args.profile, args.output, args.resume, args.stop_after, args.device, args.methods)
        elif args.command == "baselines":
            from .baselines import run_baselines
            run_baselines(args.profile, args.output, args.resume)
        elif args.command == "evaluate":
            from .learning import evaluate
            evaluate(args.profile, args.output, args.resume, args.device)
        elif args.command == "aggregate":
            from .analysis import aggregate
            aggregate(args.profile, args.output)
        elif args.command == "export-paper":
            from .export import export_paper
            export_paper(args.profile, args.output, args.paper, args.draft)
        elif args.command == "run":
            from .data import validate_data
            from .learning import train, evaluate
            from .baselines import run_baselines
            from .analysis import aggregate
            validate_data()
            train(args.profile, args.output, args.resume, device=args.device)
            run_baselines(args.profile, args.output, args.resume)
            evaluate(args.profile, args.output, args.resume, args.device)
            aggregate(args.profile, args.output)
    except (ValueError, FileNotFoundError, ImportError) as error:
        parser.exit(2, f"industrial_case: {error}\n")


if __name__ == "__main__":
    main()
