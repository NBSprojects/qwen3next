from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration du modèle (partagée entre train et sampling)."""
    n_layers: int = 6
    emb_dim: int = 768
    num_groups: int = 4
    heads_per_group: int = 4
    active_experts: int = 2
    total_experts: int = 8
    tie_embeddings: bool = False
    
    # Tokenizer
    gpt2_model_name: str = "gpt2"