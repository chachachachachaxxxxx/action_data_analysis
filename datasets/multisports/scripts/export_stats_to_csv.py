#!/usr/bin/env python3
import os
import json
import csv
from typing import Any, List

ROOT = "/storage/wangxinxing/code/action_data_analysis"
ANALYSIS_DIR = os.path.join(ROOT, "output", "MultiSports_analysis")
STATS_JSON = os.path.join(ANALYSIS_DIR, "stats.json")


def main() -> None:
    if not os.path.exists(STATS_JSON):
        raise SystemExit(f"stats.json not found: {STATS_JSON}")

    with open(STATS_JSON, "r", encoding="utf-8") as f:
        stats = json.load(f)

    # action_distribution: List[[name, count]]
    action_dist: List[List[Any]] = stats.get("action_distribution", [])
    tubes_per_action: List[List[Any]] = stats.get("spatiotemporal_tubes", {}).get("per_action", [])

    # Export class distribution
    cls_csv = os.path.join(ANALYSIS_DIR, "class_distribution.csv")
    with open(cls_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["action", "count"])
        for name, cnt in action_dist:
            w.writerow([name, int(cnt)])

    # Export tube distribution
    tubes_csv = os.path.join(ANALYSIS_DIR, "tube_distribution.csv")
    with open(tubes_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["action", "tube_count"])
        for name, cnt in tubes_per_action:
            w.writerow([name, int(cnt)])

    print("Wrote:")
    print("-", cls_csv)
    print("-", tubes_csv)


if __name__ == "__main__":
    main()
