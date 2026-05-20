"""
train.py — Fine-Tune ClimateBERT for ESG Greenwashing Detection
================================================================
Capstone Project 2025-26  |  Apoorva Singh  |  PRN 202201070030

Architecture Context
--------------------
``climatebert/distilroberta-base-climate-f`` is a DistilRoBERTa model
pre-trained on 2 million+ climate and ESG-related paragraphs
(Webersinke et al., 2022).  Unlike generic RoBERTa:

  • Its tokenizer has a domain-specific vocabulary — ESG terms like
    "Scope 3", "TCFD", and "net-zero" are recognised as single tokens
    instead of being split into sub-words, reducing information loss.
  • Its attention layers already encode semantic relationships between
    climate-adjacent concepts, so fine-tuning on our vague/specific
    labels requires far fewer examples to converge.

We add a binary classification head (linear layer → softmax) on top of
the [CLS] token embedding.  The model learns to output:
    0 → SPECIFIC  (concrete, verifiable ESG claim)
    1 → VAGUE     (hedge-laden, unverifiable language)

Class Imbalance Strategy (WeightedTrainer)
------------------------------------------
Auto-labelled data over-represents VAGUE sentences (greenwashing is
common in the wild).  ``WeightedTrainer`` applies per-sample weights
derived from inverse class frequency, so the gradient from a rare
SPECIFIC sentence is amplified to match the majority class.

Data Hierarchy
--------------
Gold labels (300 rows, manually verified) always override any matching
auto-label for the same sentence.  After deduplication, we stratify-
split into train / validation sets.

Usage
-----
    python train.py

    # Resume from a checkpoint:
    python train.py --resume models/climatebert-esg-scorer/checkpoint-500
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, cohen_kappa_score, confusion_matrix,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)
from datasets import Dataset
import evaluate
import mlflow

import config

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    """Configure root logger to write to both console and a rotating file."""
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FMT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.LOG_FILE, mode="a", encoding="utf-8"),
        ],
    )
    return logging.getLogger("train")


logger = setup_logging()


# ─────────────────────────────────────────────────────────────────────────────
# Device detection
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Automatically select the best available compute device.

    Priority: CUDA (NVIDIA GPU) → MPS (Apple Silicon) → CPU.
    Logs the selected device and, for CUDA, the GPU model name.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        logger.info("CUDA GPU detected: %s — training on GPU.", gpu_name)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Apple MPS detected — training on Apple Silicon GPU.")
    else:
        device = torch.device("cpu")
        logger.warning(
            "No GPU detected — training on CPU.  "
            "Consider reducing BATCH_SIZE and NUM_EPOCHS in config.py."
        )
    return device


# ─────────────────────────────────────────────────────────────────────────────
# Data loading & merging
# ─────────────────────────────────────────────────────────────────────────────

def load_and_merge_datasets() -> pd.DataFrame:
    dfs = []

    # 1. Load Gold Labels (The 300 Manual Rows)
    if config.GOLD_LABEL_PATH.exists():
        gold = pd.read_csv(config.GOLD_LABEL_PATH)
        
        # FIX: Rename 'manual_label' to 'label' to match config.LABEL_COL
        # Also ensure 'text_chunk' is renamed if your config says 'text'
        gold = gold.rename(columns={
            'manual_label': config.LABEL_COL,
            'text_chunk': config.TEXT_COL
        })
        
        # Only keep rows that actually have a label
        gold = gold.dropna(subset=[config.LABEL_COL])
        gold[config.SOURCE_COL] = "gold"
        
        logger.info("Loaded %d gold-labelled rows.", len(gold))
        dfs.append(gold)

    # 2. Load Auto Labels (The Full Dataset)
    if config.AUTO_LABEL_PATH.exists():
        auto = pd.read_csv(config.AUTO_LABEL_PATH)
        
        # FIX: Rename 'auto_label' to 'label' to match config.LABEL_COL
        auto = auto.rename(columns={
            'auto_label': config.LABEL_COL,
            'text_chunk': config.TEXT_COL
        })
        
        # Only keep rows where the rule-based labeller found a 0 or 1
        auto = auto.dropna(subset=[config.LABEL_COL])
        auto[config.SOURCE_COL] = "auto"
        
        logger.info("Loaded %d auto-labelled rows.", len(auto))
        dfs.append(auto)

    if not dfs:
        raise FileNotFoundError("No data files found!")

    # Combine them
    merged = pd.concat(dfs, ignore_index=True)

    # --- THE REST OF YOUR LOGIC IS CORRECT ---
    
    # Validate (This will now pass because of the renames above)
    for col in (config.TEXT_COL, config.LABEL_COL):
        if col not in merged.columns:
            raise ValueError(f"Required column '{col}' missing from dataset. Check your CSV headers!")

    # Deduplicate (Priority: Gold > Auto)
    merged["_sort_key"] = merged[config.SOURCE_COL].map({"gold": 0, "auto": 1})
    merged = (
        merged.sort_values("_sort_key")
        .drop_duplicates(subset=[config.TEXT_COL], keep="first")
        .drop(columns=["_sort_key"])
        .reset_index(drop=True)
    )

    # Normalise labels to int
    merged[config.LABEL_COL] = merged[config.LABEL_COL].astype(int)

    logger.info("Final merged dataset: %d rows.", len(merged))
    merged.to_csv(config.MERGED_DATA_PATH, index=False)
    
    return merged

def _infer_pillar(text: str) -> str:
    """
    Assign an ESG pillar (E / S / G) to a sentence via keyword matching.

    Uses the keyword dictionaries in config.PILLAR_KEYWORDS.
    Falls back to the pillar with the most keyword hits; ties broken by
    E > S > G (environmental claims are most common in greenwashing).

    Parameters
    ----------
    text : str
        Raw sentence text.

    Returns
    -------
    str
        One of "E", "S", or "G".
    """
    text_lower = text.lower()
    hits = {
        pillar: sum(kw in text_lower for kw in kws)
        for pillar, kws in config.PILLAR_KEYWORDS.items()
    }
    best = max(hits, key=hits.get)
    return best if hits[best] > 0 else "E"  # default to E if no keywords match


# ─────────────────────────────────────────────────────────────────────────────
# Tokenisation
# ─────────────────────────────────────────────────────────────────────────────

def tokenize_dataset(df: pd.DataFrame, tokenizer) -> Dataset:
    """
    Convert a DataFrame into a HuggingFace Dataset and tokenise all rows.

    ClimateBERT is a DistilRoBERTa derivative, so it expects the standard
    RoBERTa input format: [BOS] tokens [EOS] with attention masks.
    Sentences are truncated / padded to config.MAX_SEQ_LENGTH.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at minimum columns [text, label].
    tokenizer : transformers.PreTrainedTokenizer
        Instantiated tokenizer for the base model.

    Returns
    -------
    datasets.Dataset
        Tokenised dataset with input_ids, attention_mask, and labels.
    """
    hf_ds = Dataset.from_pandas(
        df[[config.TEXT_COL, config.LABEL_COL]].rename(
            columns={config.LABEL_COL: "labels"}
        )
    )

    def _tokenize(batch):
        return tokenizer(
            batch[config.TEXT_COL],
            truncation=True,
            padding="max_length",
            max_length=config.MAX_SEQ_LENGTH,
        )

    return hf_ds.map(_tokenize, batched=True, remove_columns=[config.TEXT_COL])


# ─────────────────────────────────────────────────────────────────────────────
# WeightedTrainer — handles class imbalance
# ─────────────────────────────────────────────────────────────────────────────

class WeightedTrainer(Trainer):
    """
    Custom Trainer that applies inverse-frequency class weights to the
    cross-entropy loss.

    Motivation (Slide 12 — Data Collection)
    ----------------------------------------
    Auto-labelled data skews toward VAGUE sentences because greenwashing
    is prevalent in the wild.  Without correction, the model would learn
    to predict VAGUE for most inputs, achieving low loss but missing the
    minority SPECIFIC class — which is exactly the class that signals an
    *honest* ESG claim.

    By multiplying each sample's loss by ``weight[label]``, we make the
    gradient from a rare SPECIFIC sentence as impactful as ~N VAGUE
    sentences, where N = freq(VAGUE) / freq(SPECIFIC).

    Attributes
    ----------
    class_weights : torch.Tensor
        1-D tensor of shape (num_labels,) containing per-class weights.
    """

    def __init__(self, *args, class_weights: torch.Tensor = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Override loss computation to use weighted cross-entropy.

        If class_weights are not provided, falls back to standard CE loss.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def build_compute_metrics():
    """
    Build and return a metric computation function compatible with HuggingFace
    Trainer's ``compute_metrics`` callback.

    Metrics logged (Slide 14 — Evaluation Plan)
    --------------------------------------------
    • F1 Macro   — primary metric for sentence-level classification
    • Precision  — minimise false "SPECIFIC" calls on vague text
    • Recall     — capture all vague sentences (reduces greenwashing misses)
    • AUC-ROC    — discriminative ability across decision thresholds
    • Cohen's κ  — inter-annotator agreement proxy on model predictions

    Returns
    -------
    Callable
        Function ``compute_metrics(eval_pred)`` that returns a dict of floats.
    """

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        preds = np.argmax(logits, axis=-1)

        f1   = f1_score(labels, preds, average="macro", zero_division=0)
        prec = precision_score(labels, preds, average="macro", zero_division=0)
        rec  = recall_score(labels, preds, average="macro", zero_division=0)
        kappa = cohen_kappa_score(labels, preds)

        try:
            auc = roc_auc_score(labels, probs[:, 1])
        except ValueError:
            auc = float("nan")

        cm = confusion_matrix(labels, preds)
        logger.info("Confusion Matrix:\n%s", cm)

        # Log against evaluation targets from config
        for metric_name, target in config.EVAL_TARGETS.items():
            actual = {"f1_macro": f1, "precision": prec, "auc_roc": auc,
                      "cohens_kappa": kappa}.get(metric_name)
            if actual is not None:
                status = "✓ PASS" if actual >= target else "✗ BELOW TARGET"
                logger.info(
                    "  %-20s actual=%.4f  target=%.2f  %s",
                    metric_name, actual, target, status,
                )

        return {
            "f1_macro":     f1,
            "precision":    prec,
            "recall":       rec,
            "auc_roc":      auc,
            "cohens_kappa": kappa,
        }

    return compute_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main training pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train(resume_from_checkpoint: str | None = None) -> None:
    """
    End-to-end training pipeline for the ESG Authenticity Scorer.

    Steps
    -----
    1.  Seed everything for reproducibility.
    2.  Detect compute device (CUDA / MPS / CPU).
    3.  Load and merge gold + auto-labelled datasets.
    4.  Stratified train / validation split.
    5.  Tokenise both splits.
    6.  Compute inverse-frequency class weights.
    7.  Instantiate ClimateBERT with a binary classification head.
    8.  Train with WeightedTrainer + EarlyStopping.
    9.  Save fine-tuned model and tokenizer.
    10. Log all hyperparameters and metrics to MLflow.

    Parameters
    ----------
    resume_from_checkpoint : str | None
        Path to a Trainer checkpoint directory.  Pass None to train from scratch.
    """
    set_seed(config.RANDOM_SEED)
    device = get_device()

    logger.info("=" * 70)
    logger.info("ESG Report Authenticity Scorer — Training Pipeline")
    logger.info("Base model : %s", config.BASE_MODEL_NAME)
    logger.info("Device     : %s", device)
    logger.info("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    df = load_and_merge_datasets()

    # ── 2. Train / validation split ───────────────────────────────────────────
    train_df, val_df = train_test_split(
        df,
        test_size=config.VAL_SPLIT_RATIO,
        stratify=df[config.LABEL_COL],
        random_state=config.RANDOM_SEED,
    )
    logger.info(
        "Split: %d train  |  %d validation  (stratified by label).",
        len(train_df), len(val_df),
    )

    # ── 3. Tokenise ───────────────────────────────────────────────────────────
    logger.info("Loading tokenizer from '%s' …", config.BASE_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)

    train_ds = tokenize_dataset(train_df, tokenizer)
    val_ds   = tokenize_dataset(val_df,   tokenizer)
    logger.info("Tokenisation complete.  Max seq length = %d.", config.MAX_SEQ_LENGTH)

    # ── 4. Class weights ──────────────────────────────────────────────────────
    labels_array = train_df[config.LABEL_COL].values

    if config.CLASS_WEIGHTS is not None:
        weights = torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float)
        logger.info("Using manual class weights from config: %s", weights.tolist())
    else:
        raw_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(labels_array),
            y=labels_array,
        )
        weights = torch.tensor(raw_weights, dtype=torch.float)
        logger.info(
            "Auto-computed class weights (balanced): SPECIFIC(0)=%.4f  VAGUE(1)=%.4f",
            weights[0].item(), weights[1].item(),
        )

    # ── 5. Model ──────────────────────────────────────────────────────────────
    logger.info("Loading model from '%s' …", config.BASE_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.BASE_MODEL_NAME,
        num_labels=2,
        id2label={0: "SPECIFIC", 1: "VAGUE"},
        label2id={"SPECIFIC": 0, "VAGUE": 1},
    )
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model loaded.  Trainable parameters: %s", f"{n_params:,}")

    # ── 6. TrainingArguments ──────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(config.FINE_TUNED_MODEL_PATH),
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE * 2,
        gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        warmup_ratio=config.WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_dir=str(config.LOG_DIR / "tb"),
        logging_steps=50,
        save_total_limit=3,
        seed=config.RANDOM_SEED,
        report_to="none",   # MLflow handled separately below
        # Use bf16 if CUDA supports it, else fp32
        bf16=(device.type == "cuda" and torch.cuda.is_bf16_supported()),
        dataloader_num_workers=0,  # 0 = safe for Windows / macOS
    )
    logger.info(
        "TrainingArguments set.  Effective batch size = %d.",
        config.BATCH_SIZE * config.GRAD_ACCUM_STEPS,
    )

    # ── 7. WeightedTrainer ────────────────────────────────────────────────────
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=build_compute_metrics(),
        class_weights=weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ── 8. MLflow run ─────────────────────────────────────────────────────────
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="climatebert-finetune"):
        mlflow.log_params({
            "base_model":    config.BASE_MODEL_NAME,
            "epochs":        config.NUM_EPOCHS,
            "batch_size":    config.BATCH_SIZE,
            "lr":            config.LEARNING_RATE,
            "max_seq_len":   config.MAX_SEQ_LENGTH,
            "val_split":     config.VAL_SPLIT_RATIO,
            "class_weights": weights.tolist(),
            "train_rows":    len(train_df),
            "val_rows":      len(val_df),
        })

        # ── 9. Train ──────────────────────────────────────────────────────────
        logger.info("Starting training …")
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        logger.info("Training complete.  Total steps: %d.", train_result.global_step)

        # ── 10. Evaluate best checkpoint ──────────────────────────────────────
        eval_results = trainer.evaluate()
        logger.info("Final validation metrics:")
        for k, v in eval_results.items():
            if isinstance(v, float):
                logger.info("  %-30s %.4f", k, v)
                mlflow.log_metric(k.replace("eval_", ""), v)

        # ── 11. Save model + tokenizer ────────────────────────────────────────
        trainer.save_model(str(config.FINE_TUNED_MODEL_PATH))
        tokenizer.save_pretrained(str(config.FINE_TUNED_MODEL_PATH))
        logger.info("Fine-tuned model saved to %s.", config.FINE_TUNED_MODEL_PATH)
        mlflow.log_artifact(str(config.FINE_TUNED_MODEL_PATH))

    logger.info("MLflow run complete.  Run `mlflow ui` to inspect results.")
    logger.info("=" * 70)
    logger.info("Training finished successfully.")
    logger.info("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune ClimateBERT for ESG greenwashing detection."
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CHECKPOINT_DIR",
        help="Path to a HuggingFace Trainer checkpoint to resume training from.",
    )
    args = parser.parse_args()
    train(resume_from_checkpoint=args.resume)
