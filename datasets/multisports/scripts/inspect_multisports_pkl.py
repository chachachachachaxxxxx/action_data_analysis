#!/usr/bin/env python3
import os
import sys
import pickle
import json
import numpy as np
from collections import Counter


def safe_len(obj):
    try:
        return len(obj)
    except Exception:
        return None


def summarize_container(obj, max_items=5):
    if isinstance(obj, dict):
        keys = list(obj.keys())[:max_items]
        return {
            "type": "dict",
            "size": safe_len(obj),
            "sample_keys": keys,
            "sample_value_types": {k: type(obj[k]).__name__ for k in keys},
        }
    if isinstance(obj, (list, tuple)):
        return {
            "type": type(obj).__name__,
            "size": safe_len(obj),
            "sample_types": [type(x).__name__ for x in obj[:max_items]],
        }
    return {"type": type(obj).__name__}


def try_jsonable(x):
    try:
        json.dumps(x)
        return x
    except Exception:
        return str(x)


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    pkl_path = os.path.join(repo_root, "multisports", "annotations", "multisports_GT.pkl")
    if not os.path.exists(pkl_path):
        # fallback to absolute path expectation
        pkl_path = "/storage/wangxinxing/code/action_data_analysis/multisports/annotations/multisports_GT.pkl"
    print(f"Loading: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    print("Top-level type:", type(data).__name__)

    summary = summarize_container(data)
    print("Top-level summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # If dict with splits
    if isinstance(data, dict):
        # Print split sizes
        split_sizes = {k: safe_len(v) for k, v in data.items()}
        print("Split sizes:")
        print(json.dumps(split_sizes, ensure_ascii=False, indent=2))

        # Show train/test first entries
        for split_name in ["train_videos", "test_videos"]:
            if split_name in data:
                vids = data[split_name]
                print(f"{split_name} -> size={safe_len(vids)}; sample up to 5:")
                for vv in vids[:5]:
                    print("  ", vv)

        # Pick one split to inspect deeply
        if data:
            first_split = next(iter(data.keys()))
            split_value = data[first_split]
            print(f"Inspecting split: {first_split}")
            print(json.dumps(summarize_container(split_value), ensure_ascii=False, indent=2))

            # If split is dict of videos
            if isinstance(split_value, dict) and split_value:
                first_video = next(iter(split_value.keys()))
                video_anno = split_value[first_video]
                print(f"Example video key: {first_video}")
                if isinstance(video_anno, dict):
                    print("Video annotation keys:", list(video_anno.keys()))
                    for k, v in list(video_anno.items())[:5]:
                        print(f"  {k}: {type(v).__name__} -> {summarize_container(v)}")
                else:
                    print("Video annotation type:", type(video_anno).__name__)

            # Inspect an example basketball video in gttubes
            if "gttubes" in data:
                gtt = data["gttubes"]
                # find a basketball video key
                bb_vid_key = None
                for k in gtt.keys():
                    if isinstance(k, str) and k.startswith("basketball/"):
                        bb_vid_key = k
                        break
                if bb_vid_key:
                    print(f"Inspecting gttubes for: {bb_vid_key}")
                    bb_entry = gtt[bb_vid_key]
                    print("gttubes entry type:", type(bb_entry).__name__)
                    if isinstance(bb_entry, dict):
                        print("gttubes entry keys:", list(bb_entry.keys())[:20])
                        # Often label index to list of tubes
                        for lab_k, lab_v in list(bb_entry.items())[:3]:
                            print(f"  label {lab_k}: {type(lab_v).__name__}, size={safe_len(lab_v)}")
                            if isinstance(lab_v, (list, tuple)) and lab_v:
                                tube0 = lab_v[0]
                                print("    tube[0] type:", type(tube0).__name__)
                                if isinstance(tube0, np.ndarray):
                                    print("    tube[0] shape:", tube0.shape)
                                    # print first up to 3 rows
                                    head_rows = min(3, tube0.shape[0])
                                    print("    tube[0] head rows:")
                                    for i in range(head_rows):
                                        print("      ", tube0[i].tolist())
                                    # column-wise min/max
                                    col_mins = tube0.min(axis=0).tolist()
                                    col_maxs = tube0.max(axis=0).tolist()
                                    print("    col mins:", col_mins)
                                    print("    col maxs:", col_maxs)
                                else:
                                    print("    tube[0] summary:", summarize_container(tube0))
                                if isinstance(tube0, dict):
                                    for tk, tv in tube0.items():
                                        print(f"      {tk}: {type(tv).__name__}, summary={summarize_container(tv)}")

            # Try to discover class names and sports
            label_names = set()
            sports_prefix = Counter()

            def collect_labels(obj):
                # Heuristic: strings with 'basketball' or space-delimited class names
                if isinstance(obj, str):
                    if any(s in obj.lower() for s in ["basketball", "soccer", "volleyball", "table tennis", "badminton", "hockey", "fencing", "handball"]):
                        label_names.add(obj)
                        sports_prefix[obj.split(" ")[0].lower()] += 1
                elif isinstance(obj, dict):
                    for vv in obj.values():
                        collect_labels(vv)
                elif isinstance(obj, (list, tuple)):
                    for vv in obj:
                        collect_labels(vv)

            collect_labels(data)
            print("Discovered label names (sample up to 30):")
            for ln in sorted(list(label_names))[:30]:
                print(" -", ln)
            print("Sports prefix counts (top 10):", sports_prefix.most_common(10))

    print("Done.")


if __name__ == "__main__":
    main()

