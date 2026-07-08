"""
One-time, robust downloader for the PaliGemma reader model.

Downloads the model into the standard Hugging Face cache so training
(`from_pretrained`) loads it instantly afterwards with NO re-download.

Fixes the "hangs for hours" problem via:
  - hf_transfer (parallel connections)  -> faster, more stable
  - HF_HUB_DOWNLOAD_TIMEOUT             -> a stalled connection aborts instead of hanging
  - a resume-retry loop                 -> each retry continues from where it stopped

Usage (run once, ideally inside tmux):
    export HF_TOKEN=hf_xxx            # or `huggingface-cli login`
    pip install hf_transfer
    python -u download_model.py                 # uses reader.model_name from configs/model.yaml
    python -u download_model.py --mirror        # use hf-mirror.com if HF is throttled
    python -u download_model.py --model google/paligemma-3b-mix-448
"""
import os
import sys
import time
import argparse

# --- must be set BEFORE huggingface_hub does any downloading ---
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

# IMPORTANT: hf_transfer (the fast Rust downloader) IGNORES HF_HUB_DOWNLOAD_TIMEOUT,
# so on a flaky network a stalled connection hangs forever and the resume-retry loop
# below never fires. We therefore keep it OFF by default (reliable: stalls abort in 30s
# and resume). Pass --fast to opt back in when the network is healthy.
_WANT_FAST = "--fast" in sys.argv
if _WANT_FAST:
    try:
        import hf_transfer  # noqa: F401
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        _HF_TRANSFER = True
    except Exception:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        _HF_TRANSFER = False
else:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    _HF_TRANSFER = False

import yaml
from huggingface_hub import snapshot_download


def load_model_name(default):
    """Reads reader.model_name from configs/model.yaml so it stays in sync with training."""
    try:
        with open("./configs/model.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("reader", {}).get("model_name", default)
    except Exception:
        return default


def main():
    parser = argparse.ArgumentParser(description="One-time robust downloader for the reader model.")
    parser.add_argument("--model", default=None, help="HF model id (default: reader.model_name from config)")
    parser.add_argument("--retries", type=int, default=1000, help="Max retry attempts on network errors")
    parser.add_argument("--mirror", action="store_true", help="Use https://hf-mirror.com endpoint")
    parser.add_argument("--fast", action="store_true",
                        help="Enable hf_transfer (fast but ignores the stall-timeout; only on a good network)")
    args = parser.parse_args()

    if args.mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    model = args.model or load_model_name("google/paligemma-3b-mix-448")

    print("=" * 70)
    print(f"Model            : {model}")
    print(f"hf_transfer      : {'ON' if _HF_TRANSFER else 'OFF (pip install hf_transfer for speed)'}")
    print(f"Download timeout : {os.environ.get('HF_HUB_DOWNLOAD_TIMEOUT')}s")
    print(f"Endpoint         : {os.environ.get('HF_ENDPOINT', 'https://huggingface.co (default)')}")
    print(f"HF token set     : {'yes' if (os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')) else 'using cached login (if any)'}")
    print("=" * 70)

    attempt = 0
    while True:
        attempt += 1
        try:
            path = snapshot_download(model, max_workers=4)
            print("\n" + "=" * 70)
            print(f"DONE. Model fully cached at:\n{path}")
            print("Training will now load it from cache with no further download.")
            print("=" * 70)
            return
        except KeyboardInterrupt:
            print("\nInterrupted by user. Re-run this script later; it resumes from cache.")
            sys.exit(1)
        except Exception as e:
            msg = str(e).lower()
            # Auth / gating problems are NOT network stalls -> don't retry blindly
            if any(k in msg for k in ["gated", "401", "403", "unauthorized",
                                      "access to model", "awaiting", "authentication"]):
                print("\nACCESS/AUTH problem (not a network stall):")
                print(f"  {e}")
                print("Fix: accept the model license on its HuggingFace page, then set HF_TOKEN "
                      "(or run `huggingface-cli login`), and re-run this script.")
                sys.exit(2)
            print(f"\n[attempt {attempt}] download interrupted: {e}")
            if attempt >= args.retries:
                print("Max retries reached. Re-run the script; it resumes from where it stopped.")
                sys.exit(1)
            print("Retrying in 5s (resumes from cache)...")
            time.sleep(5)


if __name__ == "__main__":
    main()
