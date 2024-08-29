from dataclasses import dataclass
from typing import Tuple, Union

@dataclass
class UNETR2DConfig:
    in_channels: int = 3
    out_channels: int = 1
    img_size: Tuple[int, int] = (256, 256)
    feature_size: int = 16
    hidden_size: int = 768
    mlp_dim: int = 3072
    num_heads: int = 12
    pos_embed: str = "perceptron"
    norm_name: Union[Tuple, str] = "instance"
    conv_block: bool = False
    res_block: bool = True
    dropout_rate: float = 0.0
    learning_rate: float = 1e-4
    batch_size: int = 16
    num_epochs: int = 50
    weight_decay: float = 1e-5
    model_save_path: str = "unetr2d_model.pth"
    dataset_path: str = 'tumour_data_archive'
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    early_stopping_patience: int = 10
    val_split: float = 0.2
    test_split: float = 0.1
    seed: int = 42
