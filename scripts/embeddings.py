from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np


TOKEN_RE = re.compile(r"[a-z0-9_./%-]+", flags=re.IGNORECASE)
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "qwen/qwen3-embedding-4b"


def skill_entrypoint(skill_root: Path) -> Path:
    lower = skill_root / "skill.md"
    upper = skill_root / "SKILL.md"
    if lower.exists():
        return lower
    if upper.exists():
        return upper
    raise FileNotFoundError(f"Could not find skill.md or SKILL.md under {skill_root}")


def load_skill_text(skill_root: Path) -> str:
    return skill_entrypoint(skill_root).read_text(encoding="utf-8", errors="ignore")


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm
    return vector


def openrouter_embed_text(
    text: str,
    *,
    model: str = DEFAULT_OPENROUTER_MODEL,
    base_url: str = DEFAULT_OPENROUTER_BASE_URL,
    api_key_env: str = "OPENROUTER_API_KEY",
    cache_dir: Path | None = None,
) -> np.ndarray:
    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing {api_key_env}; set it before running OpenRouter embeddings.")

    cache_path: Path | None = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = hashlib.sha256(f"{base_url}\n{model}\n{text}".encode("utf-8")).hexdigest()
        cache_path = cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return _normalize(np.asarray(payload["embedding"], dtype=np.float64))

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    attempt = 0
    while True:
        attempt += 1
        try:
            response = client.embeddings.create(model=model, input=text)
            if not response.data:
                raise ValueError("No embedding data received")
            embedding = response.data[0].embedding
            if not embedding:
                raise ValueError("Empty embedding vector received")
            vector = _normalize(np.asarray(embedding, dtype=np.float64))
            break
        except Exception as exc:  # noqa: BLE001
            sleep_seconds = 15
            print(
                f"[embedding] attempt={attempt} failed model={model} "
                f"text_chars={len(text)} error={type(exc).__name__}: {exc}; "
                f"retrying_in={sleep_seconds}s",
                flush=True,
            )
            time.sleep(sleep_seconds)
    if cache_path is not None:
        cache_path.write_text(
            json.dumps({"model": model, "base_url": base_url, "embedding": vector.tolist()}),
            encoding="utf-8",
        )
    return vector


def openrouter_embed_skill(
    skill_root: Path,
    *,
    model: str = DEFAULT_OPENROUTER_MODEL,
    base_url: str = DEFAULT_OPENROUTER_BASE_URL,
    api_key_env: str = "OPENROUTER_API_KEY",
    cache_dir: Path | None = None,
) -> np.ndarray:
    return openrouter_embed_text(
        load_skill_text(skill_root),
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
        cache_dir=cache_dir,
    )


def _hash_feature(feature: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    raw = int.from_bytes(digest, byteorder="little", signed=False)
    index = raw % dim
    sign = 1.0 if ((raw >> 63) & 1) == 0 else -1.0
    return index, sign


def hash_embed_text(text: str, *, dim: int = 256, include_bigrams: bool = True) -> np.ndarray:
    """Create a deterministic local embedding for a skill text.

    This is intentionally dependency-free. It is not meant to be semantically
    perfect; it gives the bandit a stable fixed representation until an API or
    learned embedding backend is plugged in.
    """
    tokens = [match.group(0).lower() for match in TOKEN_RE.finditer(text)]
    vector = np.zeros(dim, dtype=np.float64)
    for token in tokens:
        index, sign = _hash_feature("u:" + token, dim)
        vector[index] += sign
    if include_bigrams:
        for left, right in zip(tokens, tokens[1:]):
            index, sign = _hash_feature(f"b:{left} {right}", dim)
            vector[index] += sign * 0.5
    return _normalize(vector)


def hash_embed_skill(skill_root: Path, *, dim: int = 256) -> np.ndarray:
    return hash_embed_text(load_skill_text(skill_root), dim=dim)
