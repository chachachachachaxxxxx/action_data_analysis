import os
import sys
import pickle
from collections import Counter


def safe_load_pickle(path: str):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            f.seek(0)
            # Fallback for potential py2 pickles
            return pickle.load(f, encoding="latin1")


def describe_root(obj):
    print("--- ROOT SUMMARY ---")
    print("type:", type(obj).__name__)
    try:
        print("len:", len(obj))
    except Exception:
        pass
    if isinstance(obj, dict):
        keys = list(obj.keys())
        print("num_keys:", len(keys))
        print("keys_sample:", [repr(k)[:120] for k in keys[:10]])
        if keys:
            k0 = keys[0]
            v0 = obj[k0]
            print("sample_first_key:", repr(k0)[:200])
            print("sample_first_val_type:", type(v0).__name__)
            try:
                print("sample_first_val_len:", len(v0))
            except Exception:
                pass
            try:
                print("sample_first_val_repr:", repr(v0)[:600])
            except Exception:
                pass
    elif isinstance(obj, (list, tuple)):
        print("num_items:", len(obj))
        print("head_types:", [type(x).__name__ for x in obj[:5]])
        for i, x in enumerate(obj[:3]):
            print(f"item[{i}] type=", type(x).__name__)
            print(repr(x)[:300])
    else:
        print("repr:", repr(obj)[:800])


def inspect_values(obj):
    if not isinstance(obj, dict):
        return
    val_types = Counter(type(v).__name__ for v in obj.values())
    print("value_type_counts:", dict(val_types))
    for i, k in enumerate(list(obj.keys())[:3]):
        v = obj[k]
        print(f"-- Inspect key[{i}] = {repr(k)[:120]}")
        print("   type:", type(v).__name__)
        if isinstance(v, dict):
            subkeys = list(v.keys())
            print("   num_subkeys:", len(subkeys))
            print("   subkeys_sample:", subkeys[:10])
            # If common semantic fields exist, surface them
            for field in ("label", "action", "class", "start", "end", "frames", "video", "video_path", "clip"):
                if field in v:
                    print(f"   field[{field}]=", repr(v[field])[:240])
        elif isinstance(v, (list, tuple)):
            print("   num_items:", len(v))
            if v:
                print("   first_item_type:", type(v[0]).__name__)
                print("   first_item_repr:", repr(v[0])[:300])
        else:
            print("   repr:", repr(v)[:400])


def main():
    # Allow override via argv, fallback to given dataset path
    if len(sys.argv) > 1:
        pkl_path = sys.argv[1]
    else:
        pkl_path = r"D:\\data\\raw_action_dataset\\FineSports_trimmed_video_frames\\annotations\\FineSports-GT.pkl"
    print("Loading:", pkl_path)
    obj = safe_load_pickle(pkl_path)
    describe_root(obj)
    inspect_values(obj)
    print("--- DONE ---")


if __name__ == "__main__":
    main()

