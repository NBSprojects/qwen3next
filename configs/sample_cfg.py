from dataclasses import dataclass
from typing import Optional
from configs.model_cfg import ModelConfig

@dataclass
class SampleConfig(ModelConfig):
    """Configuration pour le sampling/benchmark."""
    
    # --- Sampling ---
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9
    use_kv_cache: bool = True

    # Prompt
    prompt: str = "Once upon a time"

    # Mesure
    measure_every: int = 64

    # Checkpoint
    load_ckpt: bool = True
    ckpt_path: str = "model_train.ckpt"

    # Device
    device: str = "cuda"