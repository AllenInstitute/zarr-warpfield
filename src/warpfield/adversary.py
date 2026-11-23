"""An adversarial field for the Jacobian guarantee across chunk faces.

Every chunk is easy on its own: inside a chunk the field is smooth. All the
difficulty sits on the cells that straddle chunk faces, which two chunks code
independently.

- On every chunk face (between samples kT - 1 and kT along axis a) the normal
  component steps, so that a straddling cell's normal derivative, and with it
  its det J, is a prescribed d. Along the face d sweeps smoothly,
  d = level + A sin(...), so every margin from -A to +A occurs. The step only
  shifts the rest of the array by a slowly varying amount, so chunks stay
  smooth inside.
- A is given in units of g = 2 * face_error_bound / spacing, the largest change
  of a straddling cell's normal derivative that face-bound errors on both sides
  can cause. Cells within ~1 g of a threshold can be broken by the two chunks'
  ordinary quantization errors.
- Each face gets one level: `min_det` (cells just above it must stay above it;
  those just below must stay positive), zero (cells just above must not fold;
  those just below are folded and must stay folded), or a safe control.
- Where faces of two or three axes meet, the factors multiply, so the cells
  shared by 4 or 8 chunks are hit too.
- A smooth tangential slip across every face adds off-diagonal Jacobian terms.
  They leave det J unchanged but enlarge the worst-case bound, which shows how
  conservative that bound is.
- Hash noise of `noise * error_bound` decorrelates the quantization errors of
  neighbouring face samples. It moves a straddling cell's det J by up to
  4 * noise * g / (4 * face_error_bound / error_bound), so keep it small.

The field is an analytic function of global sample coordinates, like
`warpfield.synth`, but it targets one chunking: the faces of `chunks`.
"""

from __future__ import annotations

import numpy as np

from warpfield.synth import _hash_noise

CATEGORIES = {
    0: "inside a chunk",
    1: "above min_det, must stay above",
    2: "in (0, min_det], must stay positive",
    3: "folded, must stay folded",
    4: "safe control",
    5: "chunk edge or corner (2-3 faces)",
}
LEVELS = ("min_det", "zero", "safe")


def _face_level(a: int, kf: int, seed: int) -> str:
    h = _hash_noise(np.array([a]), np.array([kf]), seed=seed)[0]
    return LEVELS[int(np.floor((h + 1.0) * 1.5)) % 3]


def _face_det(a, kf, tang, period, amp, min_det, seed):
    level = {"min_det": min_det, "zero": 0.0, "safe": 0.5 + min_det}[_face_level(a, kf, seed)]
    ph = np.pi * (_hash_noise(np.array([a]), np.array([kf]), seed=seed + 1)[0] + 1.0)
    t = sum(x / (period * (1.0 + 0.37 * i)) for i, x in enumerate(tang))
    return level + amp * np.sin(2.0 * np.pi * t + ph)


def chunk_face_adversary(
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    *,
    spacing: float = 4.0,
    error_bound: float = 0.25,
    face_error_bound: float | None = None,
    min_det: float = 0.1,
    amplitude: float = 4.0,
    period: float = 24.0,
    shear: float = 0.3,
    noise: float = 0.05,
    seed: int = 0,
):
    """(u, labels, target).

    `u` has shape (*shape, k), float32, component c along spatial axis c.
    `labels` (cell shape: one less along each axis) is the category of every
    cell, and `target` its designed det J (NaN inside chunks). `amplitude` is
    in units of g.
    """
    k = len(shape)
    if len(chunks) != k:
        raise ValueError("chunks must give one extent per spatial axis")
    eps_face = error_bound / 4 if face_error_bound is None else face_error_bound
    amp = amplitude * 2.0 * eps_face / spacing
    idx = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")

    u = np.zeros((*shape, k), dtype=np.float64)
    # smooth background: |du/dx| <~ 0.005, so interior cells have det J ~ 1
    for c in range(k):
        for a in range(k):
            L = 40.0 + 13.0 * ((a + 2 * c) % 3)
            u[..., c] += 0.03 * spacing * np.sin(2 * np.pi * idx[a] / L + c)
    for a in range(k):
        T = chunks[a]
        tang = [idx[b] for b in range(k) if b != a]
        for kf in range(1, -(-shape[a] // T)):
            past = idx[a] >= kf * T
            d = _face_det(a, kf, tang, period, amp, min_det, seed)
            u[..., a] += np.where(past, (d - 1.0) * spacing, 0.0)
            for b in range(k):
                if b == a:
                    continue
                # varies only along the third axis (if any), so it adds J[b][c]
                # terms but never changes the diagonal
                others = [idx[c] for c in range(k) if c not in (a, b)]
                phase = 2 * np.pi * others[0] / (1.7 * period) if others else 0.0
                slip = shear * spacing * np.cos(phase + kf + b)
                u[..., b] += np.where(past, slip, 0.0)
    u += (
        noise
        * error_bound
        * np.stack([_hash_noise(*idx, seed=seed + 7 + c) for c in range(k)], axis=-1)
    )

    cells = tuple(n - 1 for n in shape)
    cidx = np.meshgrid(*(np.arange(n) for n in cells), indexing="ij")
    labels = np.zeros(cells, dtype=np.int8)
    target = np.full(cells, np.nan)
    hits = np.zeros(cells, dtype=np.int8)
    for a in range(k):
        T = chunks[a]
        tang = [cidx[b] for b in range(k) if b != a]
        for kf in range(1, -(-shape[a] // T)):
            on = cidx[a] == kf * T - 1
            d = _face_det(a, kf, tang, period, amp, min_det, seed)
            level = _face_level(a, kf, seed)
            cat = np.where(
                level == "safe",
                4,
                np.where(d > min_det, 1, np.where(d > 0, 2, 3)),
            )
            labels = np.where(on, cat, labels)
            target = np.where(on, np.where(np.isnan(target), d, target * d), target)
            hits += on
    labels = np.where(hits > 1, 5, labels).astype(np.int8)
    return u.astype(np.float32), labels, target
