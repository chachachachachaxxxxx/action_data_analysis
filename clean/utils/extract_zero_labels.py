#!/usr/bin/env python3
from pathlib import Path


def main() -> None:
    base_dir = Path("/storage/wangxinxing/code/action_data_analysis/data/MultiSports_step1_first_frame/annotations")
    input_path = base_dir / "tactical.txt"
    output_path = base_dir / "tactical_label0.txt"

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for raw_line in fin:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.rsplit(",", 1)
            if len(parts) == 2 and parts[1] == "0":
                fout.write(line + "\n")

    print(f"Saved lines with label 0 to: {output_path}")


if __name__ == "__main__":
    main()


