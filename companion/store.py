"""SQLite store for people, conversation history and memories.

Names and roles are plaintext (needed to look people up); face/voice
embeddings, message text and memory facts are Fernet-encrypted by the Vault.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .vault import Vault

SCHEMA = """
CREATE TABLE IF NOT EXISTS people (
    id        INTEGER PRIMARY KEY,
    name      TEXT NOT NULL UNIQUE,
    role      TEXT NOT NULL CHECK (role IN ('owner', 'guest')),
    face_emb  BLOB,
    voice_emb BLOB,
    created   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id        INTEGER PRIMARY KEY,
    person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    ts        REAL NOT NULL,
    role      TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    text      BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS memories (
    id        INTEGER PRIMARY KEY,
    person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    ts        REAL NOT NULL,
    fact      BLOB NOT NULL
);
"""


@dataclass
class Person:
    id: int
    name: str
    role: str
    face_emb: np.ndarray | None
    voice_emb: np.ndarray | None


@dataclass
class Message:
    role: str
    text: str
    ts: float


@dataclass
class Memory:
    id: int
    fact: str
    ts: float


class Store:
    def __init__(self, data_dir: Path, vault: Vault | None = None) -> None:
        self.vault = vault or Vault(data_dir)
        # check_same_thread=False: the camera thread reads people, the main loop writes.
        self._db = sqlite3.connect(data_dir / "companion.db", check_same_thread=False)
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(SCHEMA)

    # --- people -----------------------------------------------------------------
    def add_person(self, name: str, role: str, face_emb: np.ndarray | None = None) -> int:
        v = self.vault
        with self._db:
            cur = self._db.execute(
                "INSERT INTO people (name, role, face_emb, created) VALUES (?, ?, ?, ?)",
                (name, role, v.encrypt_vector(face_emb) if face_emb is not None else None, time.time()),
            )
        return int(cur.lastrowid)

    def set_face(self, person_id: int, emb: np.ndarray) -> None:
        with self._db:
            self._db.execute("UPDATE people SET face_emb = ? WHERE id = ?", (self.vault.encrypt_vector(emb), person_id))

    def set_voice(self, person_id: int, emb: np.ndarray) -> None:
        with self._db:
            self._db.execute("UPDATE people SET voice_emb = ? WHERE id = ?", (self.vault.encrypt_vector(emb), person_id))

    def people(self) -> list[Person]:
        rows = self._db.execute("SELECT id, name, role, face_emb, voice_emb FROM people ORDER BY id").fetchall()
        v = self.vault
        return [
            Person(
                id=r[0],
                name=r[1],
                role=r[2],
                face_emb=v.decrypt_vector(r[3]) if r[3] else None,
                voice_emb=v.decrypt_vector(r[4]) if r[4] else None,
            )
            for r in rows
        ]

    def owner(self) -> Person | None:
        return next((p for p in self.people() if p.role == "owner"), None)

    def find(self, name: str) -> Person | None:
        return next((p for p in self.people() if p.name.lower() == name.lower()), None)

    def delete_person(self, person_id: int) -> None:
        with self._db:
            self._db.execute("DELETE FROM people WHERE id = ?", (person_id,))
        self._db.execute("VACUUM")  # don't leave deleted ciphertext lying in free pages

    # --- messages ---------------------------------------------------------------
    def add_message(self, person_id: int, role: str, text: str) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO messages (person_id, ts, role, text) VALUES (?, ?, ?, ?)",
                (person_id, time.time(), role, self.vault.encrypt_text(text)),
            )

    def delete_last_exchange(self, person_id: int) -> int:
        """Delete the most recent user message and everything after it; returns rows deleted."""
        row = self._db.execute(
            "SELECT MAX(id) FROM messages WHERE person_id = ? AND role = 'user'", (person_id,)
        ).fetchone()
        if row[0] is None:
            return 0
        with self._db:
            cur = self._db.execute("DELETE FROM messages WHERE person_id = ? AND id >= ?", (person_id, row[0]))
        return cur.rowcount

    def recent_messages(self, person_id: int, limit: int = 20, since: float | None = None) -> list[Message]:
        rows = self._db.execute(
            "SELECT role, text, ts FROM messages WHERE person_id = ? AND ts >= ? ORDER BY id DESC LIMIT ?",
            (person_id, since or 0.0, limit),
        ).fetchall()
        return [Message(r[0], self.vault.decrypt_text(r[1]), r[2]) for r in reversed(rows)]

    # --- memories ---------------------------------------------------------------
    def add_memory(self, person_id: int, fact: str) -> int:
        with self._db:
            cur = self._db.execute(
                "INSERT INTO memories (person_id, ts, fact) VALUES (?, ?, ?)",
                (person_id, time.time(), self.vault.encrypt_text(fact)),
            )
        return int(cur.lastrowid)

    def memories(self, person_id: int) -> list[Memory]:
        rows = self._db.execute("SELECT id, fact, ts FROM memories WHERE person_id = ? ORDER BY id", (person_id,)).fetchall()
        return [Memory(r[0], self.vault.decrypt_text(r[1]), r[2]) for r in rows]

    def delete_memory(self, memory_id: int) -> None:
        with self._db:
            self._db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))

    def replace_memories(self, person_id: int, facts: list[str]) -> None:
        with self._db:
            self._db.execute("DELETE FROM memories WHERE person_id = ?", (person_id,))
            now = time.time()
            self._db.executemany(
                "INSERT INTO memories (person_id, ts, fact) VALUES (?, ?, ?)",
                [(person_id, now, self.vault.encrypt_text(f)) for f in facts],
            )
