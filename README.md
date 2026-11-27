# palmimo-hackathon: a personal companion

Palmimo learns its owner's face once. After that it chats, keeps conversation history and builds long-term memories, but **only with that person**. Anyone else gets a friendly head shake and no cloud call.

## How it works
- **Face gate (on-device):** OpenCV YuNet finds the face and SFace turns it into 128 numbers. That vector is compared with the enrolled owner using cosine similarity (threshold 0.40). No photos are stored.
- **Privacy rule:** speech goes to OpenAI (speech-to-text and chat) only while the owner's face is in view.
- **Memory:** every message is stored encrypted. When a session ends (owner out of view for 60 s), the LLM merges the conversation into a short list of facts about you, which is added to future prompts.
- **Encryption:** face embeddings, messages and memories are Fernet-encrypted in SQLite under `~/.local/share/palmimo-hackathon/`. The key lives in the same folder with 0600 permissions. This protects copies of the database, not someone with root on the Pi.

## Use it (Portal)
1. Set `OPENAI_API_KEY` in the Portal's environment screen.
2. Launch with **mode = enroll** and your name. Look at Palmimo while it moves its head.
3. Launch with **mode = run** and talk.
   - "What do you remember about me?" reads back your memories.
   - "Forget that" drops the last exchange.
   - "Forget everything" erases all memories.
4. **mode = forget** erases your face, history and memories.

## Develop on a laptop
```bash
uv sync
uv run pytest                                          # unit tests
uv run python main.py --mode enroll --name Jamie --no-hardware --webcam
OPENAI_API_KEY=... uv run python main.py --no-hardware --webcam   # type to talk; the webcam gates who you are
OPENAI_API_KEY=... uv run python main.py --no-hardware            # no camera; prefix a line with !stranger
```

## Layout
| File | Role |
|---|---|
| `main.py` | entry point and modes |
| `companion/companion.py` | identity gate, sessions, replies, memory commands |
| `companion/face.py` | face detection, embeddings, `PresenceTracker` |
| `companion/enroll.py` | the head-sweep enrollment ritual |
| `companion/hearing.py`, `companion/vad.py` | mic → Silero VAD → utterances (VAD adapted from Jizai's companion-realtime, Apache-2.0) |
| `companion/brain.py` | OpenAI speech-to-text, chat, memory consolidation |
| `companion/body.py` | speech, neck, gestures, face display |
| `companion/store.py`, `companion/vault.py` | encrypted SQLite store |

`TEMPLATE_GUIDE.md` is the original Portal template guide (Japanese).

## Next
- Voice verification as a second check (sherpa-onnx speaker embeddings, already installed with `palmimo-sdk[voice]`).
- Guests: "This is Sam, my friend" enrolls a guest who gets friendly chat but no access to your memories.

## Debugging
Every run logs to the Portal's log screen and to `~/.local/share/palmimo-hackathon/logs/companion.log` (rotating, 2 MB × 5, readable only by you):
```bash
ssh palmimo 'tail -f ~/.local/share/palmimo-hackathon/logs/companion.log'
```
- **Startup** prints the config, SDK version, whether the API key is set, and who is enrolled.
- **Every turn** gets a number: `turn 7: gate=owner Jamie (similarity 0.71) -> speech-to-text + chat`, then `turn 7: done, stt=0.61s chat=1.20s speak=2.30s`.
- **Presence changes** (`presence: owner -> stranger (similarity 0.18, threshold 0.40)`) and gate decisions explain why Palmimo did or didn't answer.
- **Heartbeats** every 60 s: frames checked by the face gate, and the mic's peak level (0 = dead mic, 32767 = clipping).
- **OpenAI errors** are one line saying what to fix (bad key, unknown model, no internet, out of credits).
- **Crashes in background threads** are logged as `CRITICAL crash` with a traceback, instead of failing silently.

Portal params:
- **log_level = debug** adds every face match score, motion, expression and token count. Use it to tune `face_threshold`: compare your scores with a stranger's.
- **log_text** adds what was actually said. It's off by default because conversations are private.

| Symptom | Look for |
|---|---|
| Ignores you | `speech ignored: nobody in view`, or `presence: ... stranger (similarity 0.3x)` → lower `face_threshold` or re-enroll in better light |
| Never hears anything | `no audio from the mic`, or a mic heartbeat peak near 0 → try `--no-echo-cancel` |
| Answers a stranger | debug `match:` lines with stranger scores above the threshold → raise `face_threshold` |
| Robot doesn't move | `hardware connect failed` (another app holds the servos) or `neck lock active` (the neck is too hot) |
| Slow replies | the `turn N: done` timings show whether STT, chat or speech is the slow part |
