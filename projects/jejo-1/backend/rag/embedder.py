import hashlib
import math
import os
import re
import sys
import time
from typing import List, Optional

import requests

# Ensure backend directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.config import get_settings

# Upstage API interaction
UPSTAGE_API_URL = "https://api.upstage.ai/v1/embeddings"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[0-9A-Za-z가-힣_]+", text.lower())
    if tokens:
        return tokens
    stripped = text.strip().lower()
    return [stripped] if stripped else []


def _local_hash_embedding(text: str, dim: int) -> List[float]:
    """
    Deterministic local embedding fallback.
    Uses hashed token projection so retrieval still works in offline environments.
    """
    dim = max(64, int(dim))
    vector = [0.0] * dim
    tokens = _tokenize(text)

    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
        idx_a = int.from_bytes(digest[:4], "little") % dim
        idx_b = int.from_bytes(digest[4:8], "little") % dim
        weight = 1.0 + min(len(token), 24) / 24.0
        vector[idx_a] += weight
        vector[idx_b] -= weight * 0.5

    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return vector
    return [value / norm for value in vector]


def _request_upstage_embedding(text: str, api_key: str, retries: int) -> Optional[List[float]]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "input": text,
        "model": "embedding-query",
    }

    for attempt in range(retries):
        try:
            response = requests.post(
                UPSTAGE_API_URL,
                headers=headers,
                json=payload,
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            if data and "data" in data and data["data"] and "embedding" in data["data"][0]:
                return data["data"][0]["embedding"]
            print(f"Error: Unexpected API response format: {data}")
            return None
        except requests.exceptions.HTTPError as error:
            status_code = error.response.status_code if error.response is not None else "unknown"
            print(f"HTTP error occurred (Attempt {attempt + 1}/{retries}): status={status_code}, error={error}")
            if status_code in {400, 401, 403}:
                return None
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY_SECONDS)
        except requests.exceptions.ConnectionError as error:
            print(f"Connection error occurred (Attempt {attempt + 1}/{retries}): {error}")
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY_SECONDS)
        except requests.exceptions.Timeout:
            print(f"Request timed out (Attempt {attempt + 1}/{retries}).")
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY_SECONDS)
        except requests.exceptions.RequestException as error:
            print(f"Unexpected request error (Attempt {attempt + 1}/{retries}): {error}")
            return None
        except Exception as error:
            print(f"Unexpected error while calling embedding API: {error}")
            return None

    print(f"Failed to get embedding after {retries} attempt(s).")
    return None


def get_embedding(text: str) -> Optional[List[float]]:
    """
    Retrieve embedding for text.
    Priority:
    1) Upstage API (when configured and reachable)
    2) Local deterministic hash embedding fallback
    """
    if not text or not text.strip():
        print("Warning: Attempted to get embedding for empty or whitespace-only text. Returning None.")
        return None

    settings = get_settings()
    provider = str(getattr(settings, "embedding_provider", "local") or "local").strip().lower()
    if provider not in {"local", "upstage", "auto"}:
        provider = "local"

    local_dim = int(getattr(settings, "local_embedding_dim", 4096) or 4096)
    api_key = str(getattr(settings, "upstage_api_key", "") or "").strip()

    use_upstage = provider in {"upstage", "auto"} and bool(api_key)
    if use_upstage:
        retries = MAX_RETRIES if provider == "upstage" else 1
        embedding = _request_upstage_embedding(text=text, api_key=api_key, retries=retries)
        if embedding is not None:
            return embedding
        print("Warning: Upstage embedding failed. Falling back to local embedding.")
    elif provider in {"upstage", "auto"} and not api_key:
        print("Warning: UPSTAGE_API_KEY is empty. Falling back to local embedding.")

    return _local_hash_embedding(text=text, dim=local_dim)


if __name__ == "__main__":
    test_text = "안녕하세요, 무역 코칭 플랫폼입니다."
    embedding = get_embedding(test_text)
    if embedding:
        print(
            f"Embedding for '{test_text[:20]}...' "
            f"(length={len(embedding)}, first5={embedding[:5]})"
        )
    else:
        print(f"Failed to get embedding for '{test_text[:20]}...'.")
