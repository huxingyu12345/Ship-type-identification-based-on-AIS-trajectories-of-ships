from dataclasses import dataclass, field
from multiprocessing import cpu_count
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SHIP_TYPES = ["干散货船", "渔船", "油船", "集装箱船"]
SPLIT_TYPES = ["1month", "2month", "4month", "8month"]
ID_COLS = ["shipno", "shiptype", "month_split", "split_type"]
TRAJECTORY_COLS = ["shipno", "postime", "lon", "lat", "cog", "sog"]


def default_n_jobs() -> int:
    return max(1, min(10, (cpu_count() or 2) // 2))


@dataclass
class Config:
    train_dir: Path = PROJECT_ROOT / "train"
    test_dir: Path = PROJECT_ROOT / "test"
    output_dir: Path = PROJECT_ROOT / "outputs"

    num_splits: int = 8
    n_jobs: int = field(default_factory=default_n_jobs)
    img_size: tuple = (224, 224)
    image_batch_size: int = 16
    rfe_n_features: int = 60
    image_top_k: int = 20
    n_folds: int = 5
    batch_size: int = 64
    epochs: int = 100
    early_stopping_patience: int = 20

    xgb_n_estimators: int = 300
    xgb_max_depth: int = 3
    xgb_learning_rate: float = 0.01
    xgb_n_jobs: int = 3
    xgb_tree_method: str = "gpu_hist"
    xgb_gpu_id: int = 1

    @property
    def train_split_dir(self) -> Path:
        return self.output_dir / "train_split"

    @property
    def train_images_dir(self) -> Path:
        return self.output_dir / "train_images"

    @property
    def test_images_dir(self) -> Path:
        return self.output_dir / "test_images"

    @property
    def features_dir(self) -> Path:
        return self.output_dir / "features"

    @property
    def train_features_path(self) -> Path:
        return self.features_dir / "train_features.csv"

    @property
    def test_features_path(self) -> Path:
        return self.features_dir / "test_features.csv"

    @property
    def result_path(self) -> Path:
        return self.output_dir / "result.csv"

    @property
    def train_log_path(self) -> Path:
        return self.output_dir / "logs" / "train.log"

    @property
    def split_log_path(self) -> Path:
        return self.output_dir / "logs" / "split.log"

    def ensure_dirs(self) -> None:
        for path in (
            self.output_dir,
            self.train_split_dir,
            self.train_images_dir,
            self.test_images_dir,
            self.features_dir,
            self.train_log_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def max_split_id(self, split_type: str) -> int:
        if split_type == "1month":
            return self.num_splits
        if split_type == "2month":
            return self.num_splits // 2
        if split_type == "4month":
            return self.num_splits // 4
        return 1
