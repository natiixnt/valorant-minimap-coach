"""
Local callout analyzer using a fine-tuned SmolVLM model.

Drop-in replacement for AIAnalyzer that runs entirely offline
after training with scripts/train.py.

Activate in config.yaml:
    ai:
      use_local_model: true
      local_model_path: "models/minimap_coach_lora/best"
      local_model_confidence_threshold: 0.0  # always use local (no fallback)

The model is loaded lazily on first call, so startup is not slowed down.
"""
import time
from typing import TYPE_CHECKING, List, Optional

import cv2

from src.capture.screen import MinimapFrame

if TYPE_CHECKING:
    from src.telemetry.collector import DataCollector


_UNCHANGED_FALLBACK = 10.0


class LocalAnalyzer:
    """
    Identical public interface to AIAnalyzer.
    Uses a LoRA-merged SmolVLM model for inference.
    Falls back to None (silent skip) if model fails to load.
    """

    def __init__(self, config: dict, collector: "Optional[DataCollector]" = None):
        ai_cfg = config.get("ai", {})
        self.interval          = ai_cfg.get("analyze_interval", 3.0)
        self._model_path       = ai_cfg.get("local_model_path", "models/minimap_coach_lora/best")
        self._last_call:       float = 0.0
        self._last_api_call:   float = 0.0
        self._last_state:      tuple = ()
        self._last_sample_ts:  int   = 0
        self._collector                    = collector
        self._processor                    = None   # lazy
        self._model                        = None   # lazy
        self._device:          str         = "cpu"
        self._load_failed:     bool        = False

    def should_analyze(self) -> bool:
        return time.time() - self._last_call >= self.interval

    def _load(self) -> bool:
        """Lazy-load the model on first inference call."""
        if self._model is not None:
            return True
        if self._load_failed:
            return False
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForVision2Seq, AutoProcessor

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype        = torch.bfloat16 if self._device == "cuda" else torch.float32
            print(f"[LocalAnalyzer] Loading model from {self._model_path} on {self._device}...")

            self._processor = AutoProcessor.from_pretrained(self._model_path)
            base_name       = self._processor.name_or_path  # stored in config.json by save_pretrained
            base_model      = AutoModelForVision2Seq.from_pretrained(base_name, torch_dtype=dtype)
            self._model     = PeftModel.from_pretrained(base_model, self._model_path)
            self._model.eval().to(self._device)
            print(f"[LocalAnalyzer] Model ready.")
            return True
        except Exception as e:
            print(f"[LocalAnalyzer] Failed to load model: {e}")
            self._load_failed = True
            return False

    def analyze(
        self,
        frame:            MinimapFrame,
        enemy_count:      int,
        map_name:         str = "unknown",
        active_abilities: Optional[List[str]] = None,
        spike_active:     bool = False,
        recent_callouts:  Optional[List[str]] = None,
    ) -> Optional[str]:
        if not self.should_analyze():
            return None

        now = time.time()
        self._last_call = now

        state = (enemy_count, frozenset(active_abilities or []), spike_active)
        if state == self._last_state and now - self._last_api_call < _UNCHANGED_FALLBACK:
            return None

        if not self._load():
            return None

        self._last_api_call = now
        self._last_state    = state

        try:
            from PIL import Image
            import torch

            img_rgb = cv2.cvtColor(frame.data, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            parts = [f"Valorant minimap, map {map_name}, {enemy_count} enemies."]
            if spike_active:
                parts.append("Spike is planted.")
            if active_abilities:
                parts.append(f"Active: {', '.join(active_abilities)}.")
            if recent_callouts:
                parts.append(f"Recent: {'; '.join(recent_callouts[-2:])}.")
            parts.append("One new tactical callout max 10 words (threat + location). Callout only.")
            prompt = " ".join(parts)

            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]}]
            text   = self._processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self._processor(text=text, images=[pil_img], return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            try:
                with torch.no_grad():
                    out_ids = self._model.generate(
                        **inputs,
                        max_new_tokens=30,
                        do_sample=False,
                        pad_token_id=self._processor.tokenizer.pad_token_id,
                    )
            except RuntimeError as oom:
                if "out of memory" not in str(oom).lower():
                    raise
                print("[LocalAnalyzer] CUDA OOM -- falling back to CPU")
                torch.cuda.empty_cache()
                self._device = "cpu"
                self._model = self._model.to("cpu")
                inputs = {k: v.to("cpu") for k, v in inputs.items()}
                with torch.no_grad():
                    out_ids = self._model.generate(
                        **inputs,
                        max_new_tokens=30,
                        do_sample=False,
                        pad_token_id=self._processor.tokenizer.pad_token_id,
                    )

            # Decode only the generated tokens (strip the input)
            generated = out_ids[0][inputs["input_ids"].shape[1]:]
            callout   = self._processor.tokenizer.decode(generated, skip_special_tokens=True).strip()

            self._last_sample_ts = int(now)
            return callout or None
        except Exception as e:
            print(f"[LocalAnalyzer] Inference error: {e}")
            return None
