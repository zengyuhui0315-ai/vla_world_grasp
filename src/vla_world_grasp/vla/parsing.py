"""Parsing helpers for optional VLA adapters."""

from __future__ import annotations

import json
import re
from typing import Any


JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def robust_json_parse(text: str) -> Any:
    """Parse pure JSON or JSON wrapped in a Markdown code fence."""

    stripped = text.strip()
    fenced = JSON_FENCE_RE.search(stripped)
    if fenced:
        stripped = fenced.group(1).strip()
    return json.loads(stripped)
