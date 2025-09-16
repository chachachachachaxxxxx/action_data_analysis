from __future__ import annotations

import argparse
from pathlib import Path

from action_data_analysis.convert.label_map_converter import convert_dataset_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert labels in dataset JSONs using CSV mapping")
    parser.add_argument("dataset_dir", type=str, help="Path to dataset directory containing JSONs")
    parser.add_argument("mapping_csv", type=str, help="CSV file with columns label,label2")
    parser.add_argument("--output", type=str, default=None, help="Output directory (default: *_converted)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite JSON files in-place")
    parser.add_argument("--copy-images", action="store_true", help="Copy referenced images to output dir")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    stats = convert_dataset_dir(
        dataset_dir=args.dataset_dir,
        mapping_csv=args.mapping_csv,
        output_dir=args.output,
        overwrite=args.overwrite,
        copy_images=args.copy_images,
    )
    print(
        f"Done. Files: {stats.converted_files}/{stats.total_files}, Skipped: {stats.skipped_files}, "
        f"Annotations: {stats.total_annotations_before} -> {stats.total_annotations_after}, "
        f"Images copied: {stats.images_copied}"
    )


if __name__ == "__main__":
    main()


