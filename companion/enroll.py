"""Enrollment ritual: Palmimo looks at you from several angles and learns your face.

Only the averaged 128-number embedding is kept (encrypted); no photo is stored.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from .body import Body
from .face import FaceEngine, cosine, normalize
from .store import Store

log = logging.getLogger(__name__)

#: (pitch, yaw) neck poses swept during enrollment, normalized.
POSES = [(0.0, 0.0), (0.0, -0.35), (0.0, 0.35), (-0.3, 0.0), (0.2, 0.0), (0.0, 0.0)]
SAMPLES_PER_POSE = 3
MIN_SAMPLES = 8
#: A sample this far from the average is probably someone else / a bad crop.
OUTLIER_SIM = 0.5


def collect_embeddings(body: Body, camera, engine: FaceEngine) -> list[np.ndarray]:
    samples: list[np.ndarray] = []
    for pitch, yaw in POSES:
        body.look(pitch, yaw, seconds=0.8)
        time.sleep(0.3)  # let the image settle after the neck stops
        got = 0
        deadline = time.monotonic() + 2.5
        while got < SAMPLES_PER_POSE and time.monotonic() < deadline:
            frame = camera.latest(timeout=1.0)
            hit = engine.largest_face(frame) if frame is not None else None
            if hit is not None and hit.score >= 0.85:
                samples.append(hit.embedding)
                got += 1
            time.sleep(0.25)
        log.info("pose pitch=%.2f yaw=%.2f: %d samples", pitch, yaw, got)
    return samples


def average_embedding(samples: list[np.ndarray]) -> tuple[np.ndarray, int]:
    """Mean of the samples after dropping outliers; returns (embedding, samples kept)."""
    mean = normalize(np.mean(samples, axis=0))
    kept = [s for s in samples if cosine(s, mean) >= OUTLIER_SIM]
    if not kept:
        return mean, 0
    return normalize(np.mean(kept, axis=0)), len(kept)


def enroll(body: Body, camera, engine: FaceEngine, store: Store, name: str, role: str = "owner") -> bool:
    existing_owner = store.owner()
    if role == "owner" and existing_owner and existing_owner.name.lower() != name.lower():
        log.warning("enroll refused: owner is already %s", existing_owner.name)
        body.say(f"I already belong to {existing_owner.name}. Run forget first if you want a new owner.")
        return False

    body.expression("SURPRISED", 1500)
    body.say(f"Hi! I'm Palmimo. Let's get to know each other, {name}. Look at my face, and keep looking while I move my head.")
    log.info("enroll: %s as %s, sweeping %d poses", name, role, len(POSES))
    samples = collect_embeddings(body, camera, engine)
    body.look_center()
    log.info("enroll: collected %d face samples (need %d)", len(samples), MIN_SAMPLES)
    if len(samples) < MIN_SAMPLES:
        log.warning("enroll failed: too few samples (face not found, too small, or blurry; see debug log)")
        body.expression("SAD", 2000)
        body.say("Hmm, I couldn't see you well enough. Move a little closer, find some light, and let's try again.")
        return False

    emb, kept = average_embedding(samples)
    sims = sorted(cosine(s, emb) for s in samples)
    # These numbers say how consistent your samples are; live matches usually land a bit lower.
    log.info("enroll: kept %d/%d samples; similarity to average min=%.2f median=%.2f max=%.2f",
             kept, len(samples), sims[0], sims[len(sims) // 2], sims[-1])
    others = [p for p in store.people() if p.face_emb is not None and p.name.lower() != name.lower()]
    for o in others:
        log.info("enroll: similarity to already-enrolled %s = %.2f (should be well below the threshold)",
                 o.name, cosine(emb, o.face_emb))
    if kept < MIN_SAMPLES:
        log.warning("enroll failed: samples disagree (more than one face, or bad crops)")
        body.say("I saw more than one face. Let's try again with just you in front of me.")
        return False

    person = store.find(name)
    if person:
        store.set_face(person.id, emb)
    else:
        store.add_person(name, role, emb)
    log.info("enrolled %s (%s) from %d samples", name, role, kept)
    body.expression("HAPPY", 3000)
    body.nod()
    body.say(f"Nice to meet you, {name}! I'll remember your face. From now on I'm all yours.")
    return True
