#!/usr/bin/env python3
"""Check Hugging Face auth and access needed for TRIBE v2 text inference."""

from __future__ import annotations

import os
import sys


TRIBE_REPO = "facebook/tribev2"
TEXT_BACKBONE_REPO = "meta-llama/Llama-3.2-3B"


def load_local_env() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    load_dotenv()


def check_repo(api, repo_id: str, token: str) -> bool:
    try:
        info = api.model_info(repo_id, token=token)
    except Exception as exc:
        print(f"[fail] {repo_id}: {exc}")
        return False

    sha = getattr(info, "sha", None)
    gated = getattr(info, "gated", None)
    private = getattr(info, "private", None)
    print(f"[ok] {repo_id}: gated={gated} private={private} sha={sha}")
    return True


def main() -> int:
    load_local_env()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set.")
        print("Export it first, for example:")
        print("  export HF_TOKEN=hf_xxx")
        return 1

    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        print(f"Failed to import huggingface_hub: {exc}")
        print("Install dependencies first inside the venv.")
        return 1

    api = HfApi()

    try:
        who = api.whoami(token=token)
    except Exception as exc:
        print(f"[fail] token authentication: {exc}")
        return 1

    name = who.get("name") or who.get("fullname") or "<unknown>"
    print(f"[ok] token authentication: logged in as {name}")

    tribe_ok = check_repo(api, TRIBE_REPO, token)
    llama_ok = check_repo(api, TEXT_BACKBONE_REPO, token)

    if tribe_ok and llama_ok:
        print("All required Hugging Face checks passed for TRIBE v2 text inference.")
        return 0

    print("One or more Hugging Face checks failed.")
    print(f"Make sure your account can access https://huggingface.co/{TEXT_BACKBONE_REPO}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
