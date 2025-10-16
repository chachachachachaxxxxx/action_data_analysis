import argparse
import os
import sys
import re
from typing import List, Tuple

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm


def infer_device(user_device: str | None) -> str:
    if user_device:
        return user_device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def list_images_from_dir(directory: str, patterns: str) -> List[str]:
    # patterns: comma-separated globs
    import glob

    found: List[str] = []
    for pattern in [p.strip() for p in patterns.split(",")]:
        # Ensure recursive pattern
        if "**/" not in pattern:
            pattern = f"**/{pattern}"
        found.extend(glob.glob(os.path.join(directory, pattern), recursive=True))
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for p in found:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


def build_messages(image_path: str) -> list:
    prompt = (
        "你是一个篮球视频镜头分析模型，需要判断给定的画面是否为战术视角（Tactical View）。 "
        "请根据以下标准进行判断：\n\n"
        "【战术视角定义】\n"
        "当镜头以 中高俯视（约25–60°） 的角度拍摄，并能让观察者清晰辨识球员之间的空间关系、阵型结构或战术意图"
        "（如挡拆、联防、拉开空间等）时，即视为战术视角。\n\n"
        "【判断要点】\n"
        "输出 1 的条件（战术视角）：\n"
        "相机角度为 中高俯视（约25–60°）；\n"
        "画面中可辨识至少两名球员的空间站位或相互关系；\n"
        "至少满足以下任意一项：\n"
        "1.可见关键场地区域线条组合（如罚球区长方形 + 罚球弧、底线 + 三分线弧段、半场线等）；\n"
        "2.画面稳定，无明显跟拍或变焦；\n"
        "3.能推断出阵型或战术事件（如挡拆、联防、快攻布局等）；\n"
        "无论画面显示区域大小，只要空间结构与战术意图清晰，均判为战术视角。\n\n"
        "输出 0 的条件（非战术视角）：\n"
        "角度接近水平（几乎看不到地板透视）；\n"
        "画面仅聚焦单个球员或局部动作，无法读出空间关系；\n"
        "镜头频繁跟随球移动或快速变焦；\n"
        "球场线条或阵型结构无法识别。\n\n"
        "【示例】\n"
        "我给出了三张图，第一张图是正确的战术视角图，第二张图是非战术视角图，现在需要你判断第三张图\n"
        "【输出要求】\n"
        "仅输出一个数字：\n"
        "输出 1 表示该镜头属于战术视角；\n"
        "输出 0 表示该镜头不是战术视角。"
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/storage/wangxinxing/data/MultiSports_step1_first_frame/example/1.jpg"},
                {"type": "image", "image": "/storage/wangxinxing/data/MultiSports_step1_first_frame/example/0.jpg"},
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def parse_label_from_output(output_text: str) -> int:
    # Prefer the first single digit 0 or 1 in the output
    m = re.search(r"\b([01])\b", output_text)
    if m:
        return int(m.group(1))
    # Fallback: look for digits anywhere
    m = re.search(r"([01])", output_text)
    if m:
        return int(m.group(1))
    # Heuristic fallback
    normalized = output_text.strip().lower()
    if "1" in normalized or "是" in normalized:
        return 1
    return 0


def load_model_and_processor(model_path: str, device: str):
    # Choose dtype conservatively for the device
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


@torch.inference_mode()
def infer_one(model: Qwen2_5_VLForConditionalGeneration, processor: AutoProcessor, device: str, image_path: str, max_new_tokens: int) -> int:
    messages = build_messages(image_path)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    output_text = output_texts[0] if output_texts else ""
    return parse_label_from_output(output_text)


def main():
    parser = argparse.ArgumentParser(description="Batch tactical-view classification using Qwen2.5-VL")
    parser.add_argument("--input", required=True, help="目录路径")
    parser.add_argument("--output", required=True, help="输出txt文件路径，格式: file_id,label")
    parser.add_argument(
        "--pattern",
        default="**/*.jpg,**/*.jpeg,**/*.png",
        help="当输入为目录时用于匹配图片的逗号分隔glob模式",
    )
    parser.add_argument("--model", default="/storage/wangxinxing/model/Qwen/Qwen2.5-VL-7B-Instruct", help="模型/处理器路径")
    parser.add_argument("--device", default=None, help="设备: cuda/cpu，留空自动")
    parser.add_argument("--max-new-tokens", type=int, default=16, dest="max_new_tokens")
    args = parser.parse_args()

    input_path: str = args.input
    output_txt: str = args.output
    model_path: str = args.model
    device: str = infer_device(args.device)

    # Prepare items (file_id, image_path)
    items: List[Tuple[str, str]]
    if os.path.isdir(input_path):
        images = list_images_from_dir(input_path, args.pattern)
        items = [
            (os.path.splitext(os.path.basename(p))[0], p)
            for p in images
        ]
    else:
        print(f"输入为文件路径: {input_path}")
        sys.exit(1)

    if not items:
        print("未找到任何输入图片。", file=sys.stderr)
        sys.exit(1)

    print(f"加载模型与处理器自: {model_path}，设备: {device}")
    model, processor = load_model_and_processor(model_path, device)

    os.makedirs(os.path.dirname(os.path.abspath(output_txt)), exist_ok=True)

    total = len(items)
    with open(output_txt, "w", encoding="utf-8") as fout:
        for file_id, image_path in tqdm(items, desc="处理图片", unit="张"):
            try:
                label = infer_one(
                    model=model,
                    processor=processor,
                    device=device,
                    image_path=image_path,
                    max_new_tokens=args.max_new_tokens,
                )
            except Exception as e:
                # Fail-safe: record 0 for failures
                print(f"[WARN] 推理失败: {image_path} -> {e}", file=sys.stderr)
                label = 0
            fout.write(f"{file_id},{label}\n")

    print(f"完成。结果已写入: {output_txt}")


if __name__ == "__main__":
    main()


