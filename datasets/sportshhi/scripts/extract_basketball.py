import os
import pickle
from pathlib import Path
from typing import Dict, Set, Tuple

import pandas as pd


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
INPUT_DIR = os.path.join(BASE_DIR, "annotations")
OUTPUT_DIR = os.path.join(BASE_DIR, "basketball_annotations")


CSV_COLUMNS = [
    "video_id",
    "frame_id",
    "x1",
    "y1",
    "x2",
    "y2",
    "x1_2",
    "y1_2",
    "x2_2",
    "y2_2",
    "action_id",
    "person_id",
    "instance_id",
]


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_action_list(pbtxt_path: str) -> Dict[int, str]:
    id_to_name: Dict[int, str] = {}
    if not os.path.exists(pbtxt_path):
        return id_to_name
    current = {}
    with open(pbtxt_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("name:"):
                parts = line.split('"')
                current["name"] = parts[1] if len(parts) > 1 else line.split(":", 1)[1].strip()
            elif line.startswith("label_id:"):
                try:
                    current["id"] = int(line.split(":", 1)[1].strip())
                except Exception:
                    continue
            elif line.startswith("}"):
                if "id" in current and "name" in current:
                    id_to_name[current["id"]] = current["name"]
                current = {}
    if current and "id" in current and "name" in current and current["id"] not in id_to_name:
        id_to_name[current["id"]] = current["name"]
    return id_to_name


def get_basketball_action_ids(id_to_name: Dict[int, str]) -> Set[int]:
    basketball_ids: Set[int] = set()
    for action_id, name in id_to_name.items():
        if isinstance(name, str) and name.lower().startswith("basketball"):
            basketball_ids.add(action_id)
    return basketball_ids


def filter_csv(in_path: str, out_path: str, basketball_ids: Set[int]) -> Set[Tuple[str, str]]:
    if not os.path.exists(in_path):
        return set()
    # 以字符串读取以尽量保持原始格式
    df = pd.read_csv(in_path, header=None, names=CSV_COLUMNS, dtype=str)
    # 过滤 action_id 属于篮球类别
    df_filtered = df[df["action_id"].astype(int).isin(basketball_ids)]
    # 写出为无表头 CSV
    df_filtered.to_csv(out_path, index=False, header=False)
    # 返回该 split 的 (video_id, frame_id) 键集合，用于过滤 PKL
    key_pairs = set(zip(df_filtered["video_id"].tolist(), df_filtered["frame_id"].tolist()))
    return key_pairs


def filter_pkl(in_path: str, out_path: str, allowed_pairs: Set[Tuple[str, str]]) -> int:
    if not os.path.exists(in_path):
        return 0
    with open(in_path, "rb") as f:
        data = pickle.load(f)
    kept = {}
    if isinstance(data, dict):
        for k, v in data.items():
            # key 格式: "video_id,frame_id"
            if not isinstance(k, str) or "," not in k:
                continue
            vid, fid = k.split(",", 1)
            if (vid, fid) in allowed_pairs:
                kept[k] = v
    with open(out_path, "wb") as f:
        pickle.dump(kept, f)
    return len(kept)


def write_basketball_pbtxt(in_pbtxt: str, out_pbtxt: str, id_to_name: Dict[int, str]) -> int:
    basketball_items = [(i, n) for i, n in id_to_name.items() if isinstance(n, str) and n.lower().startswith("basketball")]
    # 简单写回，保留原 label_id
    with open(out_pbtxt, "w", encoding="utf-8") as f:
        for action_id, name in sorted(basketball_items, key=lambda x: x[0]):
            f.write("label {\n")
            f.write(f"  name: \"{name}\"\n")
            f.write(f"  label_id: {action_id}\n")
            f.write("  label_type: PERSON_INTERACTION\n")
            f.write("}\n")
    return len(basketball_items)


def main():
    ensure_output_dir()

    # 路径
    train_csv_in = os.path.join(INPUT_DIR, "sports_train_v1.csv")
    val_csv_in = os.path.join(INPUT_DIR, "sports_val_v1.csv")
    action_pbtxt_in = os.path.join(INPUT_DIR, "sports_action_list.pbtxt")

    train_csv_out = os.path.join(OUTPUT_DIR, "sports_train_v1.csv")
    val_csv_out = os.path.join(OUTPUT_DIR, "sports_val_v1.csv")
    action_pbtxt_out = os.path.join(OUTPUT_DIR, "sports_action_list.pbtxt")

    # 解析类别，获取篮球 action_id 列表
    id_to_name = parse_action_list(action_pbtxt_in)
    basketball_ids = get_basketball_action_ids(id_to_name)

    # 过滤 CSV，收集 frame 键，用于过滤 pkl
    train_pairs = filter_csv(train_csv_in, train_csv_out, basketball_ids)
    val_pairs = filter_csv(val_csv_in, val_csv_out, basketball_ids)

    # 写出仅含篮球的 pbtxt
    num_basketball_labels = write_basketball_pbtxt(action_pbtxt_in, action_pbtxt_out, id_to_name)

    # 过滤 PKL：proposals 与 gt
    files = [
        ("sports_dense_proposals_train.pkl", train_pairs),
        ("sports_dense_proposals_val.pkl", val_pairs),
        ("gt_dense_proposals_train.pkl", train_pairs),
        ("gt_dense_proposals_val.pkl", val_pairs),
    ]
    for fname, allowed in files:
        in_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(OUTPUT_DIR, fname)
        kept = filter_pkl(in_path, out_path, allowed)
        print(f"{fname}: kept {kept} items")

    print(f"basketball action ids: {sorted(basketball_ids)} (total {len(basketball_ids)})")
    print(f"CSV out -> train: {train_csv_out}, val: {val_csv_out}")
    print(f"PBtxt out -> {action_pbtxt_out} (labels: {num_basketball_labels})")


if __name__ == "__main__":
    main()

