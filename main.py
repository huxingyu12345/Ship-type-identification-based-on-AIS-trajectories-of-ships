import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="船舶 AIS 轨迹船型识别")
    parser.add_argument(
        "--step",
        choices=["split", "train", "all"],
        default="all",
        help="split=按时间切分训练集; train=提特征并训练预测; all=切分(如需要)+训练",
    )
    parser.add_argument("--force-split", action="store_true", help="即使已有切分结果也重新切分")
    parser.add_argument("--num-splits", type=int, default=8, help="时间切分份数，默认 8")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    from src.pipeline import run_pipeline

    run_pipeline(step=args.step, force_split=args.force_split, num_splits=args.num_splits)
