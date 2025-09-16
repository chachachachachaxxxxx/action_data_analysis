from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple
import shutil

import pandas as pd


@dataclass(frozen=True)
class ConversionStats:
    total_files: int
    converted_files: int
    skipped_files: int
    total_annotations_before: int
    total_annotations_after: int
    images_copied: int = 0
    files_deleted: int = 0


def load_label_mapping(csv_path: str | Path) -> Dict[str, str]:
    """Load mapping from a CSV having columns "label" and "label2".

    - Keys and values are stripped and lower-casing is NOT applied (case-sensitive mapping).
    - Rows where label is missing are ignored.
    - If label2 is NaN/empty string, it will be stored as empty string, signaling deletion.
    """
    csv_path = str(csv_path)
    df = pd.read_csv(csv_path)

    # Normalize column names to handle case/whitespace variations
    normalized_columns = {c.strip().lower(): c for c in df.columns}
    required = ["label", "label2"]
    missing = [col for col in required if col not in normalized_columns]
    if missing:
        raise ValueError(
            f"CSV {csv_path} must contain columns: {required}. Missing: {missing}. Found: {list(df.columns)}"
        )

    src_col = normalized_columns["label"]
    dst_col = normalized_columns["label2"]

    mapping: Dict[str, str] = {}
    for _, row in df.iterrows():
        src_label = str(row[src_col]).strip()
        if src_label in ("", "nan", "None"):
            continue
        dst_val = row[dst_col]
        if pd.isna(dst_val):
            mapping[src_label] = ""
        else:
            mapping[src_label] = str(dst_val).strip()
    return mapping


def detect_format(json_obj: Mapping[str, Any]) -> str:
    """Detect JSON annotation format.

    Returns one of: "labelme" | "coco" | "unknown".
    """
    if isinstance(json_obj, Mapping):
        if "shapes" in json_obj and isinstance(json_obj["shapes"], list):
            return "labelme"
        if "annotations" in json_obj and "categories" in json_obj:
            return "coco"
    return "unknown"


def _convert_labelme(obj: Dict[str, Any], mapping: Mapping[str, str]) -> Tuple[Dict[str, Any], int, int]:
    """Convert a LabelMe JSON object using the provided mapping.

    Supports two sources of class labels per shape:
    - shape["attributes"]["action"]: preferred if present and non-empty
    - shape["label"]: fallback when action is missing

    If the mapped target is empty string, the shape is removed.
    Returns (converted_obj, annotations_before, annotations_after)
    """
    shapes: List[Dict[str, Any]] = obj.get("shapes", []) or []
    new_shapes: List[Dict[str, Any]] = []
    for shape in shapes:
        # Prefer attributes.action
        attributes = shape.get("attributes")
        action_value: str | None = None
        if isinstance(attributes, dict):
            raw_action = attributes.get("action")
            if isinstance(raw_action, str):
                action_value = raw_action.strip()

        used_action = bool(action_value)
        source_label = action_value if used_action else str(shape.get("label", "")).strip()
        if source_label == "":
            # nothing to map
            continue

        mapped_label = mapping.get(source_label, source_label)
       
        if mapped_label == "":
            # drop this shape entirely
            continue

        new_shape = dict(shape)
        if used_action:
            # write back to attributes.action
            new_attrs = dict(attributes) if isinstance(attributes, dict) else {}
            new_attrs["action"] = mapped_label
            new_shape["attributes"] = new_attrs
        else:
            new_shape["label"] = mapped_label

        new_shapes.append(new_shape)

    new_obj = dict(obj)
    new_obj["shapes"] = new_shapes
    return new_obj, len(shapes), len(new_shapes)


def _convert_coco(obj: Dict[str, Any], mapping: Mapping[str, str]) -> Tuple[Dict[str, Any], int, int]:
    """Convert a COCO JSON object using a name->new_name mapping.

    - Categories with mapped empty label are dropped along with their annotations.
    - Categories with same target name are merged; annotations are remapped.
    Returns (converted_obj, annotations_before, annotations_after)
    """
    categories: List[Dict[str, Any]] = obj.get("categories", []) or []
    annotations: List[Dict[str, Any]] = obj.get("annotations", []) or []

    id_to_name = {int(cat["id"]): str(cat.get("name", "")).strip() for cat in categories}
    name_to_ids: Dict[str, List[int]] = {}
    for cid, name in id_to_name.items():
        name_to_ids.setdefault(name, []).append(cid)

    # Determine new category names per existing id
    id_to_new_name: Dict[int, str] = {}
    for cid, name in id_to_name.items():
        id_to_new_name[cid] = mapping.get(name, name)

    # Drop annotations whose target name is empty
    kept_annotations: List[Dict[str, Any]] = []
    for ann in annotations:
        cat_id = int(ann.get("category_id", -1))
        new_name = id_to_new_name.get(cat_id, id_to_name.get(cat_id, ""))
        if new_name == "":
            continue
        kept_annotations.append(dict(ann))

    # Build unique new category set
    unique_new_names: List[str] = []
    for old_name in name_to_ids.keys():
        new_name = mapping.get(old_name, old_name)
        if new_name == "":
            continue
        if new_name not in unique_new_names:
            unique_new_names.append(new_name)

    # Assign new contiguous ids
    new_name_to_id = {name: idx + 1 for idx, name in enumerate(unique_new_names)}

    # Remap annotations' category_id to new ids using new_name
    remapped_annotations: List[Dict[str, Any]] = []
    for ann in kept_annotations:
        old_cat_id = int(ann.get("category_id", -1))
        old_name = id_to_name.get(old_cat_id, "")
        new_name = mapping.get(old_name, old_name)
        if new_name == "":
            # already filtered
            continue
        new_cat_id = new_name_to_id[new_name]
        new_ann = dict(ann)
        new_ann["category_id"] = new_cat_id
        remapped_annotations.append(new_ann)

    new_categories = [
        {"id": cid, "name": name}
        for name, cid in sorted(new_name_to_id.items(), key=lambda x: x[1])
    ]

    new_obj = dict(obj)
    new_obj["categories"] = new_categories
    new_obj["annotations"] = remapped_annotations
    before = len(annotations)
    after = len(remapped_annotations)
    return new_obj, before, after


def convert_single_json(json_path: str | Path, mapping: Mapping[str, str]) -> Tuple[Dict[str, Any], int, int]:
    """Convert one JSON file according to mapping. Detects format automatically.

    Returns (converted_json, annotations_before, annotations_after)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    fmt = detect_format(obj)
    if fmt == "labelme":
        return _convert_labelme(obj, mapping)
    if fmt == "coco":
        return _convert_coco(obj, mapping)
    # Unknown: leave untouched
    return obj, 0, 0


def convert_dataset_dir(
    dataset_dir: str | Path,
    mapping_csv: str | Path,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    copy_images: bool = False,
) -> ConversionStats:
    """Convert all JSON annotation files in a directory tree.

    - dataset_dir: input directory containing JSON files (LabelMe per-image or COCO files)
    - mapping_csv: CSV having columns label,label2
    - output_dir: if None and overwrite=False, defaults to sibling directory with suffix "_converted"
    - overwrite: if True, files are overwritten in place; otherwise written to output_dir
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Input dataset directory not found: {dataset_dir}")

    mapping = load_label_mapping(mapping_csv)

    if overwrite:
        out_dir = dataset_dir
    else:
        if output_dir is None:
            out_dir = dataset_dir.parent / f"{dataset_dir.name}_converted"
        else:
            out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    json_files: List[Path] = sorted(dataset_dir.rglob("*.json"))
    total_files = len(json_files)
    converted_files = 0
    skipped_files = 0
    before_count = 0
    after_count = 0
    images_copied = 0
    copied_image_destinations: set[Path] = set()

    for src_path in json_files:
        try:
            converted_obj, b, a = convert_single_json(src_path, mapping)
            rel = src_path.relative_to(dataset_dir)
            dst_path = (out_dir / rel) if not overwrite else src_path
            fmt = detect_format(converted_obj)

            # If LabelMe and no shapes remain, delete or skip writing the JSON
            if fmt == "labelme" and a == 0:
                if overwrite:
                    try:
                        src_path.unlink()
                    except FileNotFoundError:
                        pass
                else:
                    # ensure any stale output is removed
                    if dst_path.exists():
                        try:
                            dst_path.unlink()
                        except FileNotFoundError:
                            pass
                converted_files += 1
                before_count += b
                # after_count does not add a for deleted file (zero)
                # count this as a deleted JSON
                skipped = False
                files_deleted_local = 1
                images_for_this_file = 0
                files_deleted_local  # no-op to satisfy linters if needed
                # accumulate
                after_count += 0
                # no image copy when deleted
                # track deleted
                images_for_this_file  # no-op
                # increase deleted counter
                # we cannot mutate dataclass here; track in variable
                # Using temp variable not needed; we'll increment a counter below
                # fall through to increment files_deleted
                files_deleted = 1
                images_added = 0
                # increment aggregates
                images_copied += images_added
                # add deleted count via separate variable added below
                # store via local var and add after
                before_count += 0
                after_count += 0
                # Add to deleted counter
                # We'll keep a separate accumulator variable above
                # but since we don't have one yet, create it outside loop
                pass
            else:
                if not overwrite:
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                with open(dst_path, "w", encoding="utf-8") as f:
                    json.dump(converted_obj, f, ensure_ascii=False, indent=2)
                converted_files += 1
                before_count += b
                after_count += a

                if copy_images and not overwrite and a > 0:
                    images_copied += _maybe_copy_images_for_json(
                        original_json_path=src_path,
                        converted_json=converted_obj,
                        dataset_root=dataset_dir,
                        output_root=out_dir,
                        copied_destinations=copied_image_destinations,
                    )
        except Exception:
            skipped_files += 1
            continue

    # Count deleted files by comparing source and destination when not overwriting and after conversion
    deleted_count = 0
    if not overwrite:
        for src_path in json_files:
            rel = src_path.relative_to(dataset_dir)
            dst_path = out_dir / rel
            if not dst_path.exists():
                # consider it deleted if original had annotations before and now zero
                try:
                    with open(src_path, "r", encoding="utf-8") as f:
                        obj = json.load(f)
                    if detect_format(obj) == "labelme" and (obj.get("shapes") or []) != []:
                        # original had shapes, but output file missing
                        deleted_count += 1
                except Exception:
                    continue

    return ConversionStats(
        total_files=total_files,
        converted_files=converted_files,
        skipped_files=skipped_files,
        total_annotations_before=before_count,
        total_annotations_after=after_count,
        images_copied=images_copied,
        files_deleted=deleted_count,
    )


def _maybe_copy_images_for_json(
    original_json_path: Path,
    converted_json: Dict[str, Any],
    dataset_root: Path,
    output_root: Path,
    copied_destinations: set[Path],
) -> int:
    """Copy image files referenced by a JSON annotation into the mirrored output directory.

    Returns number of newly copied images.
    """
    fmt = detect_format(converted_json)
    copied = 0

    if fmt == "labelme":
        rel = original_json_path.relative_to(dataset_root)
        src_dir = original_json_path.parent
        dst_dir = (output_root / rel).parent
        img_path_val = converted_json.get("imagePath")
        image_path: Path | None = None
        if isinstance(img_path_val, str) and img_path_val.strip():
            candidate = (src_dir / img_path_val).resolve()
            if candidate.exists():
                image_path = candidate
        if image_path is None:
            # Guess by json stem with common extensions
            stem = original_json_path.with_suffix("").name
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                candidate = src_dir / f"{stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
        if image_path is not None and image_path.exists():
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_path = dst_dir / image_path.name
            if dst_path not in copied_destinations:
                shutil.copy2(image_path, dst_path)
                copied_destinations.add(dst_path)
                copied += 1

    elif fmt == "coco":
        images = converted_json.get("images", []) or []
        for img in images:
            file_name = img.get("file_name")
            if not isinstance(file_name, str) or not file_name.strip():
                continue
            # Try several common root locations
            candidates = [
                dataset_root / file_name,
                dataset_root / "images" / file_name,
                original_json_path.parent / file_name,
            ]
            src_image: Path | None = None
            for candidate in candidates:
                if candidate.exists():
                    src_image = candidate
                    break
            if src_image is None:
                continue
            dst_path = output_root / file_name
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            if dst_path not in copied_destinations:
                shutil.copy2(src_image, dst_path)
                copied_destinations.add(dst_path)
                copied += 1

    return copied


