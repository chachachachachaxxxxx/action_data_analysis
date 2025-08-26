# 标准数据集格式（骨架）

- 目标：统一原始动作识别数据至标准 JSON，再可导出 CSV。

## JSON 结构（StandardDataset）

- version: string
- records: StandardRecord[]
- metadata: Metadata

StandardRecord:
- video_path: string
- action_label: string
- start_frame?: int
- end_frame?: int
- metadata?: dict

Metadata:
- source_format?: string
- created_by?: string
- created_at?: string (ISO8601)
- num_items?: int
- notes?: string
- raw_summary?: dict

## CSV 结构

- 列：video_path, action_label
- 其他信息（如帧范围、元数据）建议仅保留在 JSON 中。

> 以上为骨架说明，具体字段约束与校验规则待实现时补充。