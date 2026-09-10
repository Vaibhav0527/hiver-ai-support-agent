"""
src/agent/classifier.py

LLM-based intent classifier using OpenAI's API with structured JSON output.

Design decisions:
  - Prompt is loaded from prompts/classification.txt (version-controlled).
  - Allowed intents are loaded from configs/intents.yaml (single source of truth).
  - Model config comes from configs/model.yaml.
  - API key is read from OPENAI_API_KEY env var via python-dotenv.
  - JSON mode is used to guarantee parseable output.
  - Retry with linear backoff for transient API errors.
  - Returns a ClassificationResult dataclass — no raw dict leaking out.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    intent: str
    confidence: float
    reason: str
    model: str = ""
    latency_ms: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_prompt_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _parse_response(raw: str, allowed_intents: set) -> dict:
    """
    Extracts and validates the JSON payload from the LLM response.
    Handles cases where the model wraps JSON in markdown fences.
    """
    # Strip markdown fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON from LLM: {exc}\nRaw: {raw[:200]}") from exc

    intent = data.get("intent", "").strip()
    allowed_with_fallback = allowed_intents | {"needs_clarification"}

    # Guard: reject invented intents
    if intent not in allowed_with_fallback:
        logger.warning("LLM returned unexpected intent '%s'; falling back.", intent)
        data["intent"] = "needs_clarification"
        data["reason"] = f"Model returned unsupported intent '{intent}'"

    # Guard: confidence must be numeric
    try:
        data["confidence"] = float(data.get("confidence", 0.5))
        data["confidence"] = max(0.0, min(1.0, data["confidence"]))
    except (TypeError, ValueError):
        data["confidence"] = 0.5

    data.setdefault("reason", "No reason provided")
    return data


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class LLMIntentClassifier:
    """
    Classifies customer support messages into the AppleSupport intent taxonomy
    using an OpenAI chat model.
    """

    FALLBACK = ClassificationResult(
        intent="needs_clarification",
        confidence=0.0,
        reason="All retry attempts failed",
    )

    def __init__(
        self,
        model_config_path: str = "configs/model.yaml",
        intents_config_path: str = "configs/intents.yaml",
        prompt_path: str = "prompts/classification.txt",
    ):
        # Load configs
        model_cfg = _load_yaml(model_config_path)["model"]
        retry_cfg  = _load_yaml(model_config_path).get("retry", {})
        intents_cfg = _load_yaml(intents_config_path)

        self.model_name    = model_cfg["name"]
        self.temperature   = model_cfg.get("temperature", 0.0)
        self.max_tokens    = model_cfg.get("max_tokens", 256)
        self.timeout       = model_cfg.get("request_timeout", 30)
        self.max_attempts  = retry_cfg.get("max_attempts", 3)
        self.backoff       = retry_cfg.get("backoff_seconds", 2)

        self.allowed_intents = {i["name"] for i in intents_cfg["intents"]}
        self.intent_list_str = "\n".join(
            f"  - {i['name']}: {i['definition']}"
            for i in intents_cfg["intents"]
        )
        self.prompt_template = _load_prompt_template(prompt_path)

        # Lazy-import openai so the module is importable even without the package
        try:
            from dotenv import load_dotenv
            load_dotenv()
            import openai
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                logger.warning("OPENAI_API_KEY not set — LLM calls will fail.")
            self._client = openai.OpenAI(api_key=api_key)
            self._openai_available = True
        except ImportError:
            logger.warning("openai or dotenv not installed; LLM classifier disabled.")
            self._openai_available = False

        logger.info(
            "LLMIntentClassifier initialised | model=%s | intents=%d",
            self.model_name, len(self.allowed_intents),
        )

    # ------------------------------------------------------------------
    def classify(
        self,
        message: str,
        context: Optional[list] = None,
    ) -> ClassificationResult:
        """
        Classify a single customer message.

        Args:
            message:  The raw customer tweet/message text.
            context:  Optional list of dicts [{"role": ..., "text": ...}]
                      representing prior turns in the conversation.

        Returns:
            ClassificationResult with intent, confidence, reason.
        """
        if not self._openai_available:
            return ClassificationResult(
                intent="needs_clarification",
                confidence=0.0,
                reason="OpenAI client not available",
                error="openai package not installed",
            )

        if not message or not message.strip():
            return ClassificationResult(
                intent="needs_clarification",
                confidence=0.0,
                reason="Empty message received",
            )

        # Build context block
        context_block = ""
        if context:
            lines = [f"  [{m['role'].upper()}]: {m['text']}" for m in context]
            context_block = "\n".join(lines)
        else:
            context_block = "  (none)"

        prompt = self.prompt_template.format(
            intent_list=self.intent_list_str,
            context=context_block,
            message=message.strip(),
        )

        # Retry loop
        for attempt in range(1, self.max_attempts + 1):
            try:
                t0 = time.perf_counter()
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                    response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": prompt}],
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                raw = response.choices[0].message.content
                data = _parse_response(raw, self.allowed_intents)

                return ClassificationResult(
                    intent=data["intent"],
                    confidence=data["confidence"],
                    reason=data["reason"],
                    model=self.model_name,
                    latency_ms=round(latency_ms, 2),
                )

            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Attempt %d/%d failed: %s", attempt, self.max_attempts, exc
                )
                if attempt < self.max_attempts:
                    time.sleep(self.backoff * attempt)
                else:
                    return ClassificationResult(
                        intent="needs_clarification",
                        confidence=0.0,
                        reason="All retry attempts exhausted",
                        model=self.model_name,
                        error=str(exc),
                    )

        return self.FALLBACK
