"""On-device encryption for biometric templates and conversation data.

One Fernet key (AES-128-CBC + HMAC-SHA256) lives in the app's data directory
with 0600 permissions. It protects copies of the database (backups, a leaked
SD-card image, a stray `scp`) — not a person with root on the running Pi,
because the key sits on the same device.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from cryptography.fernet import Fernet

APP_NAME = "palmimo-hackathon"


def default_data_dir() -> Path:
    """`$XDG_DATA_HOME/palmimo-hackathon`, outside the app dir so app updates keep it."""
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / APP_NAME


class Vault:
    def __init__(self, data_dir: Path) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(data_dir, 0o700)
        key_path = data_dir / "key"
        if not key_path.exists():
            # O_EXCL: never overwrite an existing key, which would orphan all stored data.
            fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(Fernet.generate_key())
        self._fernet = Fernet(key_path.read_bytes())

    def encrypt_text(self, text: str) -> bytes:
        return self._fernet.encrypt(text.encode())

    def decrypt_text(self, token: bytes) -> str:
        return self._fernet.decrypt(token).decode()

    def encrypt_vector(self, vec: np.ndarray) -> bytes:
        return self._fernet.encrypt(np.asarray(vec, dtype=np.float32).tobytes())

    def decrypt_vector(self, token: bytes) -> np.ndarray:
        return np.frombuffer(self._fernet.decrypt(token), dtype=np.float32).copy()
