# 这个是官方的并行

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
# default: Load the model on the available device(s)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "model/Qwen2_5_VL_7B", torch_dtype="auto", device_map="auto"
# )

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/storage/wangxinxing/model/Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="cuda:0",
)

# default processer
processor = AutoProcessor.from_pretrained("/storage/wangxinxing/model/Qwen/Qwen2.5-VL-7B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
dir = '/storage/wangxinxing/code/action_data_analysis/clean/step2/test/v_It_vvQR6RPM_c005_000021.jpg'

messages1 = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/storage/wangxinxing/code/action_data_analysis/data/MultiSports_step1_first_frame/example/1.jpg",
            },
            {
                "type": "image",
                "image": "/storage/wangxinxing/code/action_data_analysis/data/MultiSports_step1_first_frame/example/0.jpg",
            },
            {
                "type": "image",
                "image": dir,
            },
            {"type": "text", "text": "你是一个篮球视频镜头分析模型，需要判断给定的画面是否为战术视角（Tactical View）。 \
请根据以下标准进行判断：\
\
【战术视角定义】\
当镜头以 中高俯视（约25–60°） 的角度拍摄，并能让观察者清晰辨识球员之间的空间关系、阵型结构或战术意图（如挡拆、联防、拉开空间等）时，即视为战术视角。\
\
【判断要点】\
 输出 1 的条件（战术视角）：\
相机角度为 中高俯视（约25–60°）；\
画面中可辨识至少两名球员的空间站位或相互关系；\
至少满足以下任意一项：\
1.可见关键场地区域线条组合（如罚球区长方形 + 罚球弧、底线 + 三分线弧段、半场线等）；\
2.画面稳定，无明显跟拍或变焦；\
3.能推断出阵型或战术事件（如挡拆、联防、快攻布局等）；\
无论画面显示区域大小，只要空间结构与战术意图清晰，均判为战术视角。\
\
输出 0 的条件（非战术视角）：\
角度接近水平（几乎看不到地板透视）；\
画面仅聚焦单个球员或局部动作，无法读出空间关系；\
镜头频繁跟随球移动或快速变焦；\
球场线条或阵型结构无法识别。\
\
【示例】\
我给出了三张图，第一张图是正确的战术视角图，第二张图是非战术视角图，现在需要你判断第三张图\
\
【输出要求】\
仅输出一个数字：\
输出 1 表示该镜头属于战术视角；\
输出 0 表示该镜头不是战术视角。"},
        ],
    }
]

messages2 = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                # "image": "/storage/wangxinxing/data/MultiSports_step1_first_frame/example/1.jpg",
                "image": "/storage/wangxinxing/code/action_data_analysis/data/MultiSports_step1_first_frame/example/1.jpg",
            },
            {
                "type": "image",
                # "image": "/storage/wangxinxing/data/MultiSports_step1_first_frame/example/0.jpg",
                "image": "/storage/wangxinxing/code/action_data_analysis/data/MultiSports_step1_first_frame/example/0.jpg",
            },
            {
                "type": "image",
                "image": dir,
            },
            {"type": "text", "text": "你是一个篮球视频镜头分析模型，需要判断给定的画面是否为战术视角（Tactical View）。 \
请根据以下标准进行判断：\
\
【战术视角定义】\
当镜头以 中高俯视（约25–60°） 的角度拍摄，并能让观察者清晰辨识球员之间的空间关系、阵型结构或战术意图（如挡拆、联防、拉开空间等）时，即视为战术视角。\
\
【判断要点】\
 输出 1 的条件（战术视角）：\
相机角度为 中高俯视（约25–60°）；\
画面中可辨识至少两名球员的空间站位或相互关系；\
至少满足以下任意一项：\
1.可见关键场地区域线条组合（如罚球区长方形 + 罚球弧、底线 + 三分线弧段、半场线等）；\
2.画面稳定，无明显跟拍或变焦；\
3.能推断出阵型或战术事件（如挡拆、联防、快攻布局等）；\
无论画面显示区域大小，只要空间结构与战术意图清晰，均判为战术视角。\
\
输出 0 的条件（非战术视角）：\
角度接近水平（几乎看不到地板透视）；\
画面仅聚焦单个球员或局部动作，无法读出空间关系；\
镜头频繁跟随球移动或快速变焦；\
球场线条或阵型结构无法识别。\
\
【示例】\
我给出了三张图，第一张图是正确的战术视角图，第二张图是非战术视角图，现在需要你判断第三张图\
\
【输出要求】\
仅输出一个数字：\
输出 1 表示该镜头属于战术视角；\
输出 0 表示该镜头不是战术视角。"},
        ],
    }
]

# 这里写一个batchsize为4的测试
# 测出来到6
batch_size = 6 
messages = [messages1] * batch_size

# Preparation for batch inference
texts = [
    processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    for msg in messages
]
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=texts,
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Batch Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_texts = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_texts)
