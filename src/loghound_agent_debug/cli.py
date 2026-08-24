"""Command-line entry point for ranking already prepared input JSONL."""

from __future__ import annotations

import argparse

from loghound_agent_debug.rankers.loghound_static import rank_file


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and delegate batch ranking to the library API."""
    parser = argparse.ArgumentParser(description="Run the static LogHound-Agent-Debug ranker")
    parser.add_argument("inputs", help="prepared inputs.jsonl")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--rules", default="configs/clue_rules.yaml")
    parser.add_argument("--output", default="outputs/rankings/rankings.jsonl")
    args = parser.parse_args(argv)
    results = rank_file(args.inputs, args.config, args.rules, args.output)
    print(f"ranked {len(results)} trajectories -> {args.output}")


if __name__ == "__main__":
    main()
