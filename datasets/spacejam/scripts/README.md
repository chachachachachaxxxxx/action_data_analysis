### scripts 目录说明（SpaceJam）

本目录用于 SpaceJam 数据的标注统计与分布导出。脚本默认使用项目内的 SpaceJam 注释与结果目录：

```text
ANNOTATION_DIR = /storage/wangxinxing/code/action_data_analysis/spacejam/annotation
RESULTS_DIR    = /storage/wangxinxing/code/action_data_analysis/spacejam/results
```

如路径不同，请在运行前修改脚本常量或通过软链接对齐。

---

#### 1) 统计划分规模、唯一性与类别分布
- 文件：`analyze_annotations.py`
- 作用：
  - 读取 `labels_dict_back.json` 的标签映射，以及 `train_back.csv`、`val_back.csv`、`test_back.csv` 的样本列表；
  - 统计各划分样本量、去重后的唯一数量、划分间交集；
  - 生成按类别的分布（各划分与整体），并写出汇总文本。
- 输入：
  - `annotation/labels_dict_back.json`（label_id -> label_name）
  - `annotation/{train_back.csv,val_back.csv,test_back.csv}`（每行形如 `path,label_id`）
- 产物（默认写入 `spacejam/results/`）：
  - `class_distribution_{train,val,test}.csv`、`class_distribution_overall.csv`
  - `split_sizes.json`、`unique_counts.json`、`intersections.json`、`labels_mapping.json`
  - `summary.txt`
- 示例：
```bash
python -m datasets.spacejam.scripts.analyze_annotations
```

---

### 推荐流程
- 准备好 SpaceJam 的标签映射与三份划分 CSV；
- 运行 `analyze_annotations.py` 生成分布与汇总；
- 依据输出的失衡度与交集情况，调整采样或划分策略。

---

### 备注
- CSV 解析对“只有一列但包含逗号”的情况做了兼容处理。
- 若需要在不同环境运行，请确保 `RESULTS_DIR` 可写，并安装标准库依赖即可。


