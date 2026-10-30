#!/usr/bin/env python3
"""Generic CO/JIT six-round repeat runner for prepare_opus_repeat plans.

Shares the already validated balanced-order timing and correctness runner.
Defaults to step2_paired/repeat_plan.json and step2_repeat.jsonl.
--dry-run validates identities without importing torch or using a GPU.
"""

from pathlib import Path

from repeat_top_candidates import main


if __name__ == "__main__":
    local = Path(__file__).resolve().parent
    main(default_plan=local / "step2_paired/repeat_plan.json",
         default_output=local / "step2_repeat.jsonl")
