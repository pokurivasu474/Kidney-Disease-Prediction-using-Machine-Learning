from dataclasses import dataclass
from config import CFG
@dataclass(frozen=True)
class Config:
    image_size: int = 320
    batch_size: int = 8
    epochs: int = 80
    learning_rate: float = 1e-4
    test_size: float = 0.2
    val_size_from_train: float = 0.1
    seed: int = 42
    threshold: float = 0.5
    image_dir: str = "/content/dataset3/image"
    mask_dir: str = "/content/dataset3/mask"


CFG = Config()