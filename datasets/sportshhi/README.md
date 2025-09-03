## SportsHHI 标注解析与结果

### 目录结构
- `annotations/`: 标注与类别文件（CSV、PKL、PBtxt）
- `scripts/`: 解析与统计脚本
- `results/`: 生成的 JSON 与说明文档

### 使用
1) 安装依赖（如需）
```
python -m pip install --user numpy pandas
```
2) 运行解析脚本
```
python sportshhi/scripts/parse_annotations.py
```
3) 查看产物
- JSON: `sportshhi/results/annotations_summary.json`
- Markdown: `sportshhi/results/README_annotations.md`

4) 转换为 LabelMe JSON（复制有标注的视频帧并生成相邻 .json）
- 一键脚本（推荐）：`sportshhi/scripts/convert_to_labelme.sh`
  - 默认路径：
    - RAW: `/storage/wangxinxing/code/action_data_analysis/data/SportsHHI/rawframes`
    - ANN: `/storage/wangxinxing/code/action_data_analysis/sportshhi/basketball_annotations`
    - OUT: `/storage/wangxinxing/code/action_data_analysis/data/SportsHHI_json`
  - 日志：写入 `OUT/convert_YYYYmmdd_HHMMSS.log`
  - 用法示例：
    - 完整转换：
      ```
      sportshhi/scripts/convert_to_labelme.sh
      ```
    - 小批量验证（3 个视频 × 每个 50 帧）：
      ```
      sportshhi/scripts/convert_to_labelme.sh --limit_videos 3 --limit_frames 50
      ```
    - 指定视频并包含副框（副框 team=1）：
      ```
      sportshhi/scripts/convert_to_labelme.sh --filter_video v_-6Os86HzwCs_c001 --include_secondary
      ```
    - 每类动作均衡采样（每类最多 20 帧）：
      ```
      sportshhi/scripts/convert_to_labelme.sh --balanced_per_action 20
      ```
  
- 直接调用 Python：`sportshhi/scripts/convert_to_labelme_json.py`
  ```
  python sportshhi/scripts/convert_to_labelme_json.py \
    --rawframes_src /path/to/rawframes \
    --ann_dir sportshhi/basketball_annotations \
    --out_root /path/to/output \
    [--limit_videos N] [--limit_frames N] \
    [--balanced_per_action N] \
    [--include_secondary] \
    [--filter_video vid1,vid2,...]
  ```
  - 说明：
    - 仅复制“有篮球标注”的视频目录。
    - 主框 `attributes.team="0"`；启用 `--include_secondary` 时，副框 `attributes.team="1"`。
    - 帧名匹配支持 `.jpg/.jpeg` 与大小写扩展名。
    - 若无法读取图像尺寸（无 Pillow/OpenCV），将以宽高 0 生成 JSON，流程不中断。

### 说明
- 坐标为归一化格式，`(x1,y1)-(x2,y2)` 为主框，`(_2)` 为副框。
- 文档包含：数据规模、类别分布、运动分布、框尺寸统计、数据质量检查、视频级 TopN。
