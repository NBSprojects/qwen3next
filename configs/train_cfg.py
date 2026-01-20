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
    batch_size: int = 128
    num_workers: int = 8

    # --- Training ---
    max_steps: int = 6_000
    eval_interval: int = 200
    log_interval: int = 20

    learning_rate: float = 3e-4
    min_lr: float = 1e-5

    # OneCycleLR parameters
    use_onecycle_scheduler: bool = True   
    pct_start: float = 0.1                # fraction du training pour le warmup (ex: 0.1 = 10%)
    anneal_strategy: str = "cos"          
    div_factor: float = 10.0              # initial_lr = max_lr / div_factor

    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: Optional[float] = 1.0
    lambda_moe: float = 0.007
    grad_log_interval: Optional[int] = 60
    activation_log_interval: Optional[int] = 60
    moe_kl_log_interval: Optional[int] = 60

    seed: int = 70
    use_moe: bool = True

    # --- Device ---
    device: str = "cuda"

    # --- Checkpoint ---
    save_model: bool = False
    save_ckpt_path: str = "model_train.ckpt"
