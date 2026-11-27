import queue

import numpy as np
import pytest

from companion.brain import parse_memories
from companion.companion import Companion, select_memories
from companion.enroll import average_embedding
from companion.face import Presence, PresenceTracker, FaceHit, normalize
from companion.store import Memory, Store
from companion.vault import Vault


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path)


def rand_emb(seed):
    return normalize(np.random.default_rng(seed).normal(size=128))


def test_vault_roundtrip_and_key_reuse(tmp_path):
    v = Vault(tmp_path)
    tok = v.encrypt_text("my sister is Ada")
    assert b"Ada" not in tok
    assert Vault(tmp_path).decrypt_text(tok) == "my sister is Ada"  # same key on reopen
    vec = rand_emb(1)
    assert np.allclose(v.decrypt_vector(v.encrypt_vector(vec)), vec)
    assert oct((tmp_path / "key").stat().st_mode & 0o777) == "0o600"


def test_store_encrypts_at_rest(tmp_path, store):
    pid = store.add_person("Jamie", "owner", rand_emb(1))
    store.add_message(pid, "user", "my sister is Ada")
    store.add_memory(pid, "Jamie's sister is Ada")
    raw = (tmp_path / "companion.db").read_bytes()
    assert b"Ada" not in raw
    assert store.owner().name == "Jamie"
    assert store.memories(pid)[0].fact == "Jamie's sister is Ada"
    assert [m.text for m in store.recent_messages(pid)] == ["my sister is Ada"]


def test_delete_person_cascades(store):
    pid = store.add_person("Jamie", "owner", rand_emb(1))
    store.add_message(pid, "user", "hi")
    store.add_memory(pid, "likes tea")
    store.delete_person(pid)
    assert store.people() == [] and store.recent_messages(pid) == [] and store.memories(pid) == []


def test_delete_last_exchange(store):
    pid = store.add_person("Jamie", "owner")
    for r, t in [("user", "a"), ("assistant", "b"), ("user", "secret"), ("assistant", "ok")]:
        store.add_message(pid, r, t)
    assert store.delete_last_exchange(pid) == 2
    assert [m.text for m in store.recent_messages(pid)] == ["a", "b"]


def test_presence_owner_stranger_and_hold(store):
    owner_emb = rand_emb(1)
    store.add_person("Jamie", "owner", owner_emb)
    t = PresenceTracker(engine=None, people_fn=store.people, threshold=0.4, hold_s=2.0)
    noisy_owner = normalize(owner_emb + 0.05 * rand_emb(9))
    t.observe(FaceHit(noisy_owner, (0.5, 0.5), 0.2, 0.9), now=100.0)
    assert t.current(now=100.1).kind == "owner"
    t.observe(None, now=101.0)  # blink / look away
    assert t.current(now=101.0).kind == "owner"  # held
    assert t.current(now=103.0).kind == "none"  # hold expired
    t.observe(FaceHit(noisy_owner, (0.5, 0.5), 0.2, 0.9), now=104.0)
    t.observe(FaceHit(rand_emb(2), (0.5, 0.5), 0.2, 0.9), now=104.2)  # someone else steps in
    assert t.current(now=104.3).kind == "stranger"  # no hold for impostors


def test_average_embedding_drops_outliers():
    base = rand_emb(1)
    samples = [normalize(base + 0.05 * rand_emb(i + 10)) for i in range(9)] + [rand_emb(99)]
    _, kept = average_embedding(samples)
    assert kept == 9


def test_parse_memories():
    assert parse_memories('{"memories": ["likes tea", " ", 3]}') == ["likes tea"]
    assert parse_memories('sure! {"memories": ["a"]}') == ["a"]
    assert parse_memories("nope") is None


def test_select_memories_small_and_large():
    small = [Memory(i, f"fact {i}", 0) for i in range(5)]
    assert select_memories(small, "x") == small
    big = [Memory(i, f"fact {i}", 0) for i in range(60)] + [Memory(60, "sister Ada lives in Lagos", 0)]
    big = [Memory(0, "sister Ada lives in Lagos", 0)] + big[1:]
    picked = select_memories(big, "how is my sister doing")
    assert len(picked) <= 35 and any("Ada" in m.fact for m in picked)


class FakeBody:
    speaking = False

    def __init__(self):
        self.said = []
        self.gestures = []

    def say(self, t): self.said.append(t)
    def expression(self, *a): pass
    def look_toward(self, c): pass
    def nod(self): self.gestures.append("nod")
    def head_shake(self): self.gestures.append("shake")
    def wave(self): self.gestures.append("wave")


class FakeBrain:
    def __init__(self):
        self.calls = []

    def transcribe(self, wav):
        self.calls.append("stt")
        return "hello"

    def reply(self, name, memories, history, text):
        self.calls.append("chat")
        return f"hi {name}, you said {text}"

    def consolidate_memories(self, name, current, session):
        self.calls.append("memory")
        return ["Jamie said hello"]


class FakeTracker:
    def __init__(self, p): self.p = p
    def current(self): return self.p


def test_gate_owner_gets_cloud_stranger_does_not(store):
    pid = store.add_person("Jamie", "owner", rand_emb(1))
    owner = store.owner()
    body, brain = FakeBody(), FakeBrain()
    tracker = FakeTracker(Presence("stranger", None, (0.5, 0.5)))
    c = Companion(body, store, brain, tracker, queue.Queue())

    c._handle("audio", b"wav")
    assert brain.calls == []  # stranger: nothing sent to OpenAI
    assert "shake" in body.gestures

    tracker.p = Presence("owner", owner, (0.5, 0.5), 0.8)
    c._tick_presence()
    c._handle("audio", b"wav")
    assert brain.calls == ["stt", "chat"]
    assert len(store.recent_messages(pid)) == 2

    c._end_session(blocking=True)
    assert [m.fact for m in store.memories(pid)] == ["Jamie said hello"]


def test_local_memory_commands(store):
    pid = store.add_person("Jamie", "owner", rand_emb(1))
    store.add_memory(pid, "Jamie likes tea")
    c = Companion(FakeBody(), store, FakeBrain(), FakeTracker(Presence("none")), queue.Queue())
    owner = store.owner()
    assert "likes tea" in c._local_command(owner, "What do you remember about me?")
    assert c._local_command(owner, "forget everything") is not None
    assert store.memories(pid) == []


def test_logs_hide_conversation_by_default(store, tmp_path, caplog):
    import logging
    from companion import logs

    logs.setup_logging(tmp_path, "debug", log_text=False)
    caplog.set_level(logging.DEBUG)
    store.add_person("Jamie", "owner", rand_emb(1))
    brain = FakeBrain()
    brain.transcribe = lambda wav: "my secret password is swordfish"
    c = Companion(FakeBody(), store, brain, FakeTracker(Presence("owner", store.owner(), (0.5, 0.5), 0.9)), queue.Queue())
    c._handle("audio", b"wav")
    assert "turn 1: gate=owner" in caplog.text
    assert "swordfish" not in caplog.text
    assert "swordfish" not in (tmp_path / "logs" / "companion.log").read_text()
    assert logs.text("abc") == "<3 chars>"
    logs.setup_logging(tmp_path, "info", log_text=True)
    assert logs.text("abc") == "'abc'"
    logs.setup_logging(tmp_path, "info", log_text=False)
