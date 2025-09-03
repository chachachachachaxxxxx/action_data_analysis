from __future__ import annotations

import argparse


def main() -> None:
  parser = argparse.ArgumentParser(
    prog="action-data-analysis",
    description="动作识别数据集分析 · CLI（骨架）",
  )
  subparsers = parser.add_subparsers(dest="command")

  # convert 子命令（骨架）
  subparsers.add_parser("convert", help="格式转换：pkl/raw -> standard.json -> csv")

  # analyze 子命令（骨架）
  subparsers.add_parser("analyze", help="分析：视觉/统计（输入 json/csv）")

  args = parser.parse_args()
  raise NotImplementedError("TODO: CLI 具体子命令逻辑与参数定义")


if __name__ == "__main__":  # pragma: no cover
  main()