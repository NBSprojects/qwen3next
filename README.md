Ce repo contient une implémentation **compacte** et pédagogique d'un LLM suivant l'architecture de Qwen3

---

## Structure du repo

- `train.py` : entraînement (WikiText-103) + logs + plots
- `benchmark_sampling.py` : génération + benchmark latence (avec/sans KV-cache)
- `model.py` : modèle `DecoderOnlyLM` (MoE) + `DecoderOnlyLMDense` (baseline dense)
- `decoder.py` : bloc decoder (Pre-Norm RMSNorm + GQA + MoE/FFN)
- `gqa.py` : attention GQA + RoPE + **KV cache**
- `MoE.py` : MoE top-k (inspiré Switch/DeepSeek) + FFN SwiGLU
- `dataprep.py` : chargement/tokenization + création de blocks
- `configs/` : configs dataclass (train / sample / modèle)


Notes :
- `torch.compile` (utilisé dans `train.py`) nécessite **PyTorch ≥ 2.0**.
- Le training force du **bfloat16** dans `train.py` (voir section “Types & dtypes”). Sur GPU non compatible bf16, il faudra adapter.

---

## Installation, Train, Sample

### 1) Installer les dépendances

```bash
pip install -r requirements.txt
```

### 2) Créer le dossier `analytics/`

Les scripts sauvegardent des figures dans `analytics/`

```bash
mkdir -p analytics
```

### 3) Lancer un entraînement

Ouvrir/éditer la config : `configs/train_cfg.py` et `configs/model_cfg.py`
Points utiles : `use_moe`, `tie_embeddings`, `active_experts`, `total_experts`
⚠️ Par défaut, `save_model=False`


```bash
python train.py
```

Sorties :
- logs dans le terminal
- figures dans :
  - `analytics/training_curves.png`
  - `analytics/gradient_norms.png`

---

### 4) Sampling

Vérifier la config : `configs/sample_cfg.py`

Points utiles : `use_kv_cache` (pour observer la différence de vitesse de sampling), `load_ckpt` (initialement à False)

```bash
python benchmark_sampling.py
```

Sorties :
- un exemple de génération dans le terminal
- un plot de latence dans `analytics/per_token_times.png`

---



