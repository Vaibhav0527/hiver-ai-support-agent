"""
src/agent/reply_generator.py

Generates grounded customer-support replies using historical evidence.

Design principles:
  - The prompt explicitly forbids the model from inventing policies, links,
    fees, or refund amounts not present in the retrieved historical examples.
  - Structured output includes:
      * reply       — the customer-facing message
      * confidence  — 0-1 float: how well-grounded in evidence the reply is
      * evidence    — list of {conversation_id, reason} showing which examples
                      were used and why
      * unsupported_claims — any claims the model flagged as not backed by evidence
  - Reuses the same retry/error-handling pattern as classifier.py.
  - Prompt is loaded from prompts/reply_generation.txt (version-controlled).
  - API key from OPENAI_API_KEY env var via python-dotenv.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class EvidenceItem:
    conversation_id: str
    reason: str


@dataclass
class ReplyResult:
    reply: str
    confidence: float
    evidence: list[EvidenceItem] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    model: str = ""
    latency_ms: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _format_context(context: list) -> str:
    if not context:
        return "  (none — this is the first message)"
    return "\n".join(
        f"  [{m.get('role','?').upper()}]: {m.get('text','')}"
        for m in context
    )


def _format_historical(examples: list) -> str:
    """
    Render retrieved examples for injection into the prompt.
    Each example shows the customer issue, the brand's response, and similarity.
    """
    if not examples:
        return "  (no similar historical examples found)"

    blocks = []
    for i, ex in enumerate(examples, 1):
        # Support both RetrievedExample dataclasses and plain dicts
        if hasattr(ex, "conversation_id"):
            cid     = ex.conversation_id
            cust    = ex.customer_message
            brand   = ex.historical_brand_response
            score   = getattr(ex, "similarity_score", "?")
            intent  = ex.intent or "unknown"
        else:
            cid   = ex.get("conversation_id", "?")
            cust  = ex.get("customer_message", "")
            brand = ex.get("historical_brand_response", "")
            score = ex.get("similarity_score", "?")
            intent = ex.get("intent", "unknown")

        blocks.append(
            f"  Example {i} [id={cid}, similarity={score}, intent={intent}]\n"
            f"    Customer: {cust}\n"
            f"    Brand reply: {brand}"
        )
    return "\n\n".join(blocks)


def _parse_response(raw: str) -> dict:
    """Extract and validate the JSON payload from the LLM response."""
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON from LLM: {exc}\nRaw: {raw[:200]}") from exc

    # Defaults + type coercion
    data.setdefault("reply", "I'm sorry, I couldn't generate a reply.")
    data.setdefault("evidence", [])
    data.setdefault("unsupported_claims", [])
    try:
        data["confidence"] = float(data.get("confidence", 0.5))
        data["confidence"] = max(0.0, min(1.0, data["confidence"]))
    except (TypeError, ValueError):
        data["confidence"] = 0.5

    # Normalise evidence list
    parsed_evidence = []
    for item in data.get("evidence", []):
        if isinstance(item, dict):
            parsed_evidence.append(
                EvidenceItem(
                    conversation_id=str(item.get("conversation_id", "")),
                    reason=str(item.get("reason", "")),
                )
            )
    data["evidence"] = parsed_evidence

    if not isinstance(data["unsupported_claims"], list):
        data["unsupported_claims"] = []

    return data


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class ReplyGenerator:
    """
    Generates grounded customer-support replies backed by historical evidence.
    """

    FALLBACK = ReplyResult(
        reply="I'm sorry, I'm unable to help with that right now. Please contact Apple Support directly at support.apple.com.",
        confidence=0.0,
        error="All retry attempts failed",
    )

    def __init__(
        self,
        model_config_path: str = "configs/model.yaml",
        prompt_path: str = "prompts/reply_generation.txt",
    ):
        model_cfg  = _load_yaml(model_config_path)["model"]
        retry_cfg  = _load_yaml(model_config_path).get("retry", {})

        self.model_name   = model_cfg["name"]
        self.temperature  = model_cfg.get("temperature", 0.0)
        self.max_tokens   = 512   # replies need more room than classification
        self.timeout      = model_cfg.get("request_timeout", 30)
        self.max_attempts = retry_cfg.get("max_attempts", 3)
        self.backoff      = retry_cfg.get("backoff_seconds", 2)

        self.prompt_template = _load_prompt(prompt_path)

        try:
            from dotenv import load_dotenv
            load_dotenv()
            import openai
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                logger.warning("OPENAI_API_KEY not set — reply generation will fail.")
            self._client = openai.OpenAI(api_key=api_key)
            self._openai_available = True
        except ImportError:
            logger.warning("openai or dotenv not installed; ReplyGenerator disabled.")
            self._openai_available = False

        logger.info("ReplyGenerator initialised | model=%s", self.model_name)

    # ------------------------------------------------------------------

    def generate(
        self,
        customer_message: str,
        intent: str,
        context: Optional[list] = None,
        historical_examples: Optional[list] = None,
    ) -> ReplyResult:
        """
        Generate a grounded support reply.

        Args:
            customer_message:   The raw customer message text.
            intent:             Classified intent label (from LLMIntentClassifier).
            context:            Prior conversation turns [{role, text}].
            historical_examples: Retrieved similar conversations (RetrievedExample objects or dicts).

        Returns:
            ReplyResult with reply, confidence, evidence, and unsupported_claims.
        """
        if not self._openai_available:
            return ReplyResult(
                reply="OpenAI client not available.",
                confidence=0.0,
                error="openai package not installed",
            )

        if not customer_message or not customer_message.strip():
            return ReplyResult(
                reply="Could you please describe your issue? I'd be happy to help!",
                confidence=0.0,
                error="empty customer message",
            )

        prompt = self.prompt_template.format(
            intent=intent or "unknown",
            context=_format_context(context or []),
            customer_message=customer_message.strip(),
            historical_examples=_format_historical(historical_examples or []),
        )

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
                raw  = response.choices[0].message.content
                data = _parse_response(raw)

                return ReplyResult(
                    reply=data["reply"],
                    confidence=data["confidence"],
                    evidence=data["evidence"],
                    unsupported_claims=data["unsupported_claims"],
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
                    return ReplyResult(
                        reply=self.FALLBACK.reply,
                        confidence=0.0,
                        model=self.model_name,
                        error=str(exc),
                    )

        return self.FALLBACK
