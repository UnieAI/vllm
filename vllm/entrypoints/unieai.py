# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import os
import base64
import json
import hashlib
import asyncio
import pathlib
import subprocess
import urllib.request
import urllib.error

from datetime import datetime, timezone

try:
    import uvloop
except ImportError:
    uvloop = None

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
except ImportError:
    print("ERROR: 'cryptography' package is required for license verification.")
    print("Install it with: pip install cryptography")
    sys.exit(1)

from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.logger import init_logger

logger = init_logger(__name__)

LICENSE_SERVER = {
    "host": [
        "https://auth.unieai.com",
        "https://uls.unieai.com",
        "https://13.114.141.202",
        "http://13.114.141.202",
    ],
    "info": """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA5Xtc4qwU2nxODyg3h2i8
2wMYofSUA9ZKpjaaLE1sbH8gJrij5KxSAShtLgq9I5O6FKtLfA+OVdYJnM7TzMS7
DIIgxabp3+x4NEbhUW2zi2Z4sX2eUFHlSDcy6xNoi6txk9KpHMKYt0QtHL7XGJPN
lULIvG5zwDTbJY3MpYiJW27U8qCbCGq/gnV3mva1NLNjL0vqTeiUQgPiwYakEPuJ
H0Yt5exYueMltRoTxRIOq2uK6KPJiQu0f9m1u/J3PXoTZN4WyySXealneN95wfeF
InxZBNLEBVHnJ1adWSAcmIdPLvljDixpMt57OPUa7dEDXO6e5mKF9aj9HcAER8BC
lQIDAQAB
-----END PUBLIC KEY-----"""
}

CACHE_DIR = pathlib.Path(os.environ.get("UNIEAI_CACHE_DIR", str(pathlib.Path.home() / ".unieai")))
CACHE_FILE = CACHE_DIR / "last_verified.json"

# ---------------------------------------------------------------------------
# Device fingerprint
# ---------------------------------------------------------------------------

def get_session_id() -> str:
    """Derive a deterministic session ID from NVIDIA GPU UUIDs.

    Runs ``nvidia-smi --query-gpu=uuid --format=csv,noheader``, collects all
    GPU UUIDs, sorts them alphabetically, concatenates them, and returns the
    SHA-256 hex digest as the session ID.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            logger.error("nvidia-smi failed (rc=%d): %s", result.returncode, result.stderr.strip())
            sys.exit(1)

        uuids = sorted(line.strip() for line in result.stdout.strip().splitlines() if line.strip())
        if not uuids:
            logger.error("nvidia-smi returned no GPU UUIDs.")
            sys.exit(1)

        combined = ",".join(uuids)
        session_id = hashlib.sha256(combined.encode("utf-8")).hexdigest()
        logger.info("Session ID derived from %d GPU(s): %s", len(uuids), session_id)
        return session_id

    except FileNotFoundError:
        logger.error("nvidia-smi not found. NVIDIA drivers must be installed.")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        logger.error("nvidia-smi timed out.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# License helpers
# ---------------------------------------------------------------------------

def _load_public_key():
    """Load the embedded RSA public key."""
    pem_data = LICENSE_SERVER["info"].encode("utf-8")
    return serialization.load_pem_public_key(pem_data)


def decrypt_license() -> dict:
    """Read UNIEAI_LICENSE env var, decrypt with public key, return license data.

    The env var holds a base64-encoded blob containing:
        {
            "data":      "<base64-encoded license JSON>",
            "signature": "<base64-encoded RSA signature of the data bytes>"
        }

    The public key is used to verify the signature (i.e. "decrypt"), proving
    the data was produced by the holder of the private key.  On success the
    inner license JSON is returned as a dict.  Expected fields:

        license_key  – e.g. "UNIE-TEST-001"
        session_id   – SHA-256 of sorted GPU UUIDs
        expires_at   – ISO-8601 date, e.g. "2026-12-31"
    """
    raw = os.environ.get("UNIEAI_LICENSE")
    if not raw:
        logger.error("UNIEAI_LICENSE environment variable is not set.")
        sys.exit(1)

    # ── Step 1: base64-decode the outer envelope ──────────────────────────
    try:
        envelope = json.loads(base64.b64decode(raw))
    except Exception as exc:
        logger.error("Failed to decode UNIEAI_LICENSE: %s", exc)
        sys.exit(1)

    data_b64: str | None = envelope.get("data")
    sig_b64: str | None = envelope.get("signature")

    if not data_b64 or not sig_b64:
        logger.error(
            "UNIEAI_LICENSE envelope must contain 'data' and 'signature' fields."
        )
        sys.exit(1)

    data_bytes = base64.b64decode(data_b64)
    sig_bytes = base64.b64decode(sig_b64)

    # ── Step 2: verify RSA signature (PSS + SHA-256) ─────────────────────
    try:
        public_key = _load_public_key()
        public_key.verify(
            sig_bytes,
            data_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        logger.info("License decrypted and verified successfully.")
    except Exception as exc:
        logger.error("License verification failed (invalid signature): %s", exc)
        sys.exit(1)

    # ── Step 3: parse the inner JSON ─────────────────────────────────────
    try:
        license_data = json.loads(data_bytes)
    except Exception as exc:
        logger.error("Failed to parse license data JSON: %s", exc)
        sys.exit(1)

    required = ("license_key", "session_id", "expires_at")
    missing = [k for k in required if k not in license_data]
    if missing:
        logger.error("License data is missing required fields: %s", missing)
        sys.exit(1)

    logger.info(
        "License loaded — key=%s, session=%s, expires=%s",
        license_data["license_key"],
        license_data["session_id"],
        license_data["expires_at"],
    )
    return license_data


# ---------------------------------------------------------------------------
# Online verification
# ---------------------------------------------------------------------------

def verify_license_online(license_key: str, session_id: str) -> bool:
    """POST heartbeat to each license server host until one succeeds.

    On success, caches the verification timestamp to ``~/.unieai/last_verified.json``.
    Returns True if the server confirms ACTIVE + allowed.
    """
    body = json.dumps({
        "license_key": license_key,
        "session_id": session_id,
    }).encode("utf-8")

    # Allow self-signed / IP-based certs
    import ssl as _ssl
    ctx = _ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = _ssl.CERT_NONE

    last_error: Exception | None = None

    for host in LICENSE_SERVER["host"]:
        url = f"{host}/api/licenses/heartbeat"
        try:
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
                resp_body = json.loads(resp.read().decode("utf-8"))

            status = resp_body.get("status", "").upper()
            allowed = resp_body.get("allowed", False)

            if status == "ACTIVE" and allowed:
                logger.info(
                    "License heartbeat OK via %s — status=%s, allowed=%s",
                    host, status, allowed,
                )
                _write_verified_cache(license_key, resp_body)
                return True
            else:
                logger.warning("License rejected by %s — %s", host, resp_body)
                return False

        except Exception as exc:
            last_error = exc
            logger.debug("Heartbeat failed for %s: %s", host, exc)
            continue

    logger.warning("All license server hosts unreachable. Last error: %s", last_error)
    return False


def _write_verified_cache(license_key: str, server_response: dict):
    """Persist the last successful verification to disk."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache = {
            "license_key": license_key,
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "server_response": server_response,
        }
        CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        logger.info("Last-verified cache written to %s", CACHE_FILE)
    except Exception as exc:
        logger.warning("Could not write verification cache: %s", exc)


def _read_verified_cache() -> dict | None:
    """Read the last-verified cache, or None if unavailable."""
    try:
        if CACHE_FILE.exists():
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read verification cache: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Validity check
# ---------------------------------------------------------------------------

def check_license_validity(license_data: dict, online_ok: bool) -> bool:
    """Decide whether the license allows the server to launch.

    * If online verification succeeded → check ``expires_at`` is in the future.
    * If all servers were unreachable → allow launch only if the cached
      last-verified timestamp exists AND ``expires_at`` is still in the future.
    """
    now = datetime.now(timezone.utc)

    # Parse expiry
    try:
        expires_str = license_data["expires_at"]
        # Support both date-only ("2026-12-31") and full ISO datetime
        if "T" in expires_str:
            expires_at = datetime.fromisoformat(expires_str)
        else:
            expires_at = datetime.fromisoformat(expires_str + "T23:59:59+00:00")
        # Ensure timezone-aware
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
    except Exception as exc:
        logger.error("Invalid expires_at in license data: %s", exc)
        return False

    if expires_at <= now:
        logger.error(
            "License has expired (expires_at=%s, now=%s).",
            expires_at.isoformat(), now.isoformat(),
        )
        return False

    if online_ok:
        logger.info("License is valid (online verified, expires %s).", expires_at.date())
        return True

    # Offline fallback — check cache
    cache = _read_verified_cache()
    if cache is None:
        logger.error(
            "All license servers are unreachable and no previous verification "
            "cache exists. Cannot launch."
        )
        return False

    logger.info(
        "All license servers are unreachable, but license was last verified "
        "at %s and does not expire until %s. Allowing launch.",
        cache.get("verified_at", "unknown"),
        expires_at.date(),
    )
    return True


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def main():
    parser = FlexibleArgumentParser(
        description="UnieAI CLI - Wrapper for vLLM activities"
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    openai_parser = subparsers.add_parser(
        "openai-api-server",
        help="Launch the OpenAI-compatible API server",
    )
    make_arg_parser(openai_parser)

    args = parser.parse_args()

    if args.subcommand == "openai-api-server":
        # ── License verification (first boot) ────────────────────────────
        logger.info("Verifying UnieAI license …")

        license_data = decrypt_license()
        device_session_id = get_session_id()

        # Verify that the license is bound to this machine
        if license_data["session_id"] != device_session_id:
            logger.error(
                "License session_id mismatch: license=%s, device=%s. "
                "This license is not valid for this machine.",
                license_data["session_id"], device_session_id,
            )
            sys.exit(1)
        logger.info("License session_id matches device fingerprint.")

        online_ok = verify_license_online(
            license_data["license_key"],
            device_session_id,
        )

        if not check_license_validity(license_data, online_ok):
            logger.error("License verification failed. Aborting server launch.")
            sys.exit(1)

        logger.info("License check passed. Starting server …")

        # ── Launch the vLLM server ───────────────────────────────────────
        validate_parsed_serve_args(args)

        if uvloop is not None:
            uvloop.run(run_server(args))
        else:
            asyncio.run(run_server(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
