from dataclasses import dataclass, field
from typing import Optional, Tuple
from configs.model_cfg import ModelConfig

@dataclass
class TrainConfig(ModelConfig):
    """Configuration pour l'entraînement."""
    
    # --- Données ---
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-v1"
    block_size: int = 256
    batch_size: int = 32
    num_workers: int = 4

    # --- Training ---
    max_steps: int = 20_000
    eval_interval: int = 200
    log_interval: int = 20

    learning_rate: float = 1e-4
    min_lr: float = 1e-5
    use_cosine_scheduler: bool = True
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: Optional[float] = 1.0
    lambda_moe: float = 0.005
    grad_log_interval: Optional[int] = 60

    seed: int = 147
    use_moe: bool = True

    # --- Device ---
    device: str = "cuda"

    # --- Checkpoint ---
    save_model: bool = False
    save_ckpt_path: str = "model_train.ckpt"