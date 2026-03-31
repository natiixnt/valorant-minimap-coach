"""
Fine-tune SmolVLM-500M on collected minimap callout data using LoRA.

This script trains a local vision-language model that can replace Claude
for AI callout generation, eliminating ongoing API costs after training.

Requirements (install separately):
    pip install -r scripts/requirements.txt

Usage:
    # Basic run (uses data/collected/minimap_callout/ by default)
    python scripts/train.py

    # Custom paths / options
    python scripts/train.py \\
        --data_dir data/collected \\
        --output_dir models/minimap_coach_lora \\
        --base_model HuggingFaceTB/SmolVLM-500M-Instruct \\
        --epochs 3 \\
        --min_samples 500

After training, update config.yaml:
    ai:
      use_local_model: true
      local_model_path: "models/minimap_coach_lora"

The local model is then used via src/vision/local_analyzer.py.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_dataset(data_dir: str, min_positive_ratio: float = 0.0):
    """
    Walk data/collected/minimap_callout/ and return list of
    {"image_path": str, "label": str, "feedback": dict} dicts.

    Samples with feedback.negative > feedback.positive are excluded when
    min_positive_ratio > 0, enabling feedback-guided quality filtering.
    """
    import json
    from pathlib import Path

    base = Path(data_dir) / "minimap_callout"
    if not base.exists():
        raise FileNotFoundError(f"No minimap_callout data found at {base}")

    samples = []
    skipped_feedback = 0

    for label_dir in sorted(base.iterdir()):
        if not label_dir.is_dir():
            continue
        meta_path = label_dir / "meta.json"
        if not meta_path.exists():
            continue

        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"Skipping {meta_path}: {e}")
            continue
        label = meta.get("label", "")
        if not label:
            continue

        fb       = meta.get("feedback", {})
        pos      = fb.get("positive", 0)
        neg      = fb.get("negative", 0)
        total_fb = pos + neg
        if total_fb > 0 and min_positive_ratio > 0:
            ratio = pos / total_fb
            if ratio < min_positive_ratio:
                skipped_feedback += 1
                continue

        for img_path in sorted(label_dir.glob("*.jpg")):
            samples.append({
                "image_path":     str(img_path),
                "label":          label,
                "map":            meta.get("map", ""),
                "spike_active":   meta.get("spike_active", False),
                "feedback_pos":   pos,
                "feedback_neg":   neg,
            })

    log.info(f"Loaded {len(samples)} samples ({skipped_feedback} label dirs skipped by feedback filter)")
    return samples


class MinimapDataset:
    def __init__(self, samples: list, processor, max_new_tokens: int = 30):
        self.samples        = samples
        self.processor      = processor
        self.max_new_tokens = max_new_tokens

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        from PIL import Image
        s     = self.samples[idx]
        try:
            image = Image.open(s["image_path"]).convert("RGB")
        except Exception as e:
            log.warning(f"Failed to load image {s['image_path']}: {e} - using blank")
            image = Image.new("RGB", (224, 224))

        parts = [f"Valorant minimap, map {s['map']}." if s["map"] else "Valorant minimap."]
        if s.get("spike_active"):
            parts.append("Spike is planted.")
        parts.append("One tactical callout max 10 words (threat + location). Callout only.")
        prompt = " ".join(parts)

        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=[image], return_tensors="pt")

        # Labels: mask the prompt tokens, only supervise the answer
        input_ids = inputs["input_ids"].squeeze(0)
        labels    = input_ids.clone()

        answer_ids = self.processor.tokenizer(
            s["label"] + self.processor.tokenizer.eos_token,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].squeeze(0)

        # Mask all but the last len(answer_ids) tokens
        labels[: len(labels) - len(answer_ids)] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values":   inputs["pixel_values"].squeeze(0),
            "labels":         labels,
        }


def collate_fn(batch: list, pad_token_id: int):
    import torch
    from torch.nn.utils.rnn import pad_sequence

    input_ids   = pad_sequence([b["input_ids"]   for b in batch], batch_first=True, padding_value=pad_token_id)
    labels      = pad_sequence([b["labels"]       for b in batch], batch_first=True, padding_value=-100)
    attn_mask   = pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0)
    pixel_values = torch.stack([b["pixel_values"] for b in batch])

    return {
        "input_ids":      input_ids,
        "attention_mask": attn_mask,
        "pixel_values":   pixel_values,
        "labels":         labels,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    data_dir:          str   = "data/collected",
    output_dir:        str   = "models/minimap_coach_lora",
    base_model:        str   = "HuggingFaceTB/SmolVLM-500M-Instruct",
    epochs:            int   = 3,
    batch_size:        int   = 4,
    lr:                float = 2e-4,
    lora_r:            int   = 16,
    lora_alpha:        int   = 32,
    min_samples:       int   = 200,
    min_positive_ratio: float = 0.4,
    val_split:         float = 0.1,
    seed:              int   = 42,
) -> None:
    import random
    import torch
    from torch.utils.data import DataLoader, random_split
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from peft import LoraConfig, get_peft_model, TaskType

    random.seed(seed)
    torch.manual_seed(seed)

    # Load data
    samples = load_dataset(data_dir, min_positive_ratio=min_positive_ratio)
    if len(samples) < min_samples:
        log.error(f"Only {len(samples)} samples, need at least {min_samples}. Collect more data first.")
        return

    # Load model + processor
    log.info(f"Loading {base_model}...")
    processor = AutoProcessor.from_pretrained(base_model)
    model     = AutoModelForVision2Seq.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    # LoRA
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.to(device)

    # Dataset split
    dataset  = MinimapDataset(samples, processor)
    val_size = max(1, int(len(dataset) * val_split))
    trn_size = len(dataset) - val_size
    trn_ds, val_ds = random_split(dataset, [trn_size, val_size],
                                  generator=torch.Generator().manual_seed(seed))

    pad_id  = processor.tokenizer.pad_token_id or 0
    collate = lambda b: collate_fn(b, pad_id)
    trn_loader = DataLoader(trn_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(trn_loader))

    # Training loop
    best_val_loss = float("inf")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(trn_loader):
            batch    = {k: v.to(device) for k, v in batch.items()}
            outputs  = model(**batch)
            loss     = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            if step % 20 == 0:
                log.info(f"Epoch {epoch} step {step}/{len(trn_loader)}  loss={loss.item():.4f}")

        avg_trn = total_loss / len(trn_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch     = {k: v.to(device) for k, v in batch.items()}
                val_loss += model(**batch).loss.item()
        avg_val = val_loss / len(val_loader)
        log.info(f"Epoch {epoch}  train={avg_trn:.4f}  val={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            model.save_pretrained(out / "best")
            processor.save_pretrained(out / "best")
            log.info(f"Saved best model (val_loss={best_val_loss:.4f})")

    # Final save
    model.save_pretrained(out / "final")
    processor.save_pretrained(out / "final")

    # Write training manifest
    manifest = {
        "base_model":    base_model,
        "samples_used":  len(samples),
        "epochs":        epochs,
        "best_val_loss": best_val_loss,
        "lora_r":        lora_r,
        "lora_alpha":    lora_alpha,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info(f"Training complete. Model saved to {out}")
    log.info("Update config.yaml:  ai.use_local_model: true  ai.local_model_path: models/minimap_coach_lora/best")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fine-tune SmolVLM on minimap callout data")
    p.add_argument("--data_dir",          default="data/collected")
    p.add_argument("--output_dir",        default="models/minimap_coach_lora")
    p.add_argument("--base_model",        default="HuggingFaceTB/SmolVLM-500M-Instruct")
    p.add_argument("--epochs",            type=int,   default=3)
    p.add_argument("--batch_size",        type=int,   default=4)
    p.add_argument("--lr",                type=float, default=2e-4)
    p.add_argument("--lora_r",            type=int,   default=16)
    p.add_argument("--min_samples",       type=int,   default=200)
    p.add_argument("--min_positive_ratio",type=float, default=0.4,
                   help="Min fraction of positive feedback to include a label dir (0=disable)")
    p.add_argument("--val_split",         type=float, default=0.1)
    args = p.parse_args()
    train(**vars(args))
