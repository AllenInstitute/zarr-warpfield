"""Procedural synthetic registration fields with known discontinuities.

Every field is an analytic function of *global* sample coordinates, so any
region of an arbitrarily large array can be generated on its own and written
chunk by chunk (`Preset.region`). Displacements are in image voxels; field
sample `i` along a spatial axis sits at image coordinate `i * spacing`.
Component `c` is the displacement along the c-th spatial axis.

Each field is the sum of:

- a smooth background: global translation and small linear term, a few
  low-frequency Fourier modes, Gaussian bumps, a mid-frequency ripple, and
  hash noise (sigma ~ 0.02 voxel) that imitates solver noise;
- *tangential slip cracks*: a jump parallel to a curved surface that ends at
  tips inside the volume (24 voxels, and a long one of 96 voxels);
- an *opening crack*: a 12-voxel jump normal to a surface, away from it (a gap);
- a *fold*: a 16-voxel jump normal to a surface, back across it, so tissue from
  the two sides overlaps (the forward map is not injective);
- a *steep compression band*: a tanh profile 1-2 samples wide that stays
  invertible (det J ~ 0.3);
- *over-compression*: the same continuous profile, strong enough that det J < 0;
- for stacked presets, *misaligned sections* (a whole-slice offset) and, for
  the 2D stack, a crack that only exists in a range of sections.

`labels` marks where each discontinuity lies (see LABELS).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LABELS = {
    0: "none",
    1: "slip crack",
    2: "opening crack",
    3: "steep band (det J > 0)",
    4: "over-compression (continuous, det J < 0)",
    5: "misaligned section",
    6: "fold (overlap jump)",
}
CRACK_LABELS = (1, 2, 6)  # labels that are true steps in the field


# -- deterministic hash noise --------------------------------------------------
def _hash_noise(*idx: np.ndarray, seed: int) -> np.ndarray:
    """Uniform-ish noise in [-1, 1) from integer coordinates (splitmix64)."""
    with np.errstate(over="ignore"):
        h = np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15)
        for a in idx:
            h = h ^ (np.asarray(a).astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15))
            h = (h ^ (h >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            h = (h ^ (h >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            h = h ^ (h >> np.uint64(31))
    return (h >> np.uint64(11)).astype(np.float64) / float(1 << 52) - 1.0


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def _perp(n: np.ndarray, rng) -> tuple[np.ndarray, ...]:
    """Orthonormal tangents to n (1 in 2D, 2 in 3D)."""
    if n.size == 2:
        return (np.array([-n[1], n[0]]),)
    a = _unit(rng.normal(size=3))
    t1 = _unit(a - n * (a @ n))
    return (t1, np.cross(n, t1))


# -- one field (2 or 3 spatial dims), optionally evolving with a parameter t ---
class Field:
    def __init__(self, ks: int, extent: np.ndarray, spacing: np.ndarray, seed: int):
        rng = np.random.default_rng(seed)
        self.ks, self.extent, self.h, self.seed = ks, extent, spacing, seed
        L = float(extent.mean())
        h = float(spacing.mean())
        # background
        self.shift = rng.uniform(-15, 15, ks)
        self.lin = rng.uniform(-0.02, 0.02, (ks, ks))
        self.modes = []
        for _ in range(5):
            lam = rng.uniform(0.5, 1.5) * L
            k = _unit(rng.normal(size=ks)) * 2 * np.pi / lam
            amp = rng.uniform(0.3, 1.0) * 0.03 / (2 * np.pi / lam) * _unit(rng.normal(size=ks))
            self.modes.append((k, amp, rng.uniform(0, 2 * np.pi), rng.uniform(-0.15, 0.15)))
        self.bumps = []
        for _ in range(3):
            c = rng.uniform(0.15, 0.85, ks) * extent
            sig = rng.uniform(0.08, 0.2) * L
            a = rng.uniform(0.02, 0.05) * sig * _unit(rng.normal(size=ks))
            self.bumps.append((c, sig, a))
        lam_mid = 12 * h
        self.ripple = (_unit(rng.normal(size=ks)) * 2 * np.pi / lam_mid, 0.25)
        self.noise = 0.02

        # Cracks are true steps: the displacement jumps by `jump` image voxels across the
        # surface g = 0 (applied on the g > 0 side, decaying away from it). The jump fades
        # to zero towards the tips over `tip` voxels, so it never needs an infinite gradient.
        def crack(lo, hi, b, lam, half, jump, decay):
            n = _unit(rng.normal(size=ks))
            c = rng.uniform(lo, hi, ks) * extent
            return dict(
                c=c,
                n=n,
                t=_perp(n, rng),
                b=b * L,
                lam=lam * L,
                half=half * L,
                jump=jump,
                decay=decay * L,
                tip=max(0.2 * half * L, 1.5 * jump),
            )

        # (direction of the jump, params, label): slip moves along the surface; open moves
        # the far side away along +n (a gap, not surjective); fold moves it back along -n,
        # so the two sides overlap (not injective) -- a section fold.
        self.cracks = [("slip", crack(0.35, 0.55, 0.04, 0.6, 0.3, 24.0, 0.6), 1)]
        self.cracks.append(("open", crack(0.55, 0.75, 0.02, 0.5, 0.22, 12.0, 0.5), 2))
        n3 = _unit(rng.normal(size=ks))
        self.band = dict(
            c=rng.uniform(0.2, 0.4, ks) * extent,
            n=n3,
            t=_perp(n3, rng),
            w=1.5 * h,
            ratio=0.7,
            sig=0.18 * L,
        )
        n4 = _unit(rng.normal(size=ks))
        self.fold = dict(
            c=rng.uniform(0.65, 0.8, ks) * extent,
            n=n4,
            t=_perp(n4, rng),
            w=1.5 * h,
            ratio=1.8,
            sig=0.08 * L,
        )
        self.cracks.append(("fold", crack(0.2, 0.35, 0.03, 0.5, 0.25, 16.0, 0.5), 6))
        self.cracks.append(("slip", crack(0.4, 0.6, 0.03, 0.9, 0.42, 96.0, 0.8), 1))

    # surfaces ---------------------------------------------------------------
    @staticmethod
    def _crack_g(P, cr, drift):
        c = cr["c"] + drift
        d = [P[i] - c[i] for i in range(len(P))]
        dn = sum(cr["n"][i] * d[i] for i in range(len(P)))
        tc = [sum(t[i] * d[i] for i in range(len(P))) for t in cr["t"]]
        wav = np.sin(2 * np.pi * tc[0] / cr["lam"])
        if len(tc) > 1:
            wav = wav * np.cos(2 * np.pi * tc[1] / cr["lam"])
        g = dn - cr["b"] * wav
        r = np.sqrt(sum(x * x for x in tc))
        taper = np.clip((cr["half"] - r) / cr["tip"], 0.0, 1.0)
        return g, taper, tc

    def _crack(self, P, cr, direction: str, drift, amp_scale):
        g, taper, _ = self._crack_g(P, cr, drift)
        side = (g > 0).astype(np.float64)
        mag = cr["jump"] * amp_scale * taper * side * np.exp(-np.maximum(g, 0) / cr["decay"])
        vec = {"slip": cr["t"][0], "open": cr["n"], "fold": -cr["n"]}[direction]
        return [mag * vec[i] for i in range(self.ks)]

    def _crack_label(self, P, cr, drift):
        g, taper, _ = self._crack_g(P, cr, drift)
        lab = np.zeros(g.shape, dtype=bool)
        for ax in range(self.ks):
            for sgn in (-1, 1):
                Q = list(P)
                Q[ax] = P[ax] + sgn * self.h[ax]
                g2, _, _ = self._crack_g(Q, cr, drift)
                lab |= (g > 0) != (g2 > 0)
        return lab & (taper > 0.05)

    def _profile(self, P, bd, drift, amp_scale):
        c = bd["c"] + drift
        d = [P[i] - c[i] for i in range(len(P))]
        s = sum(bd["n"][i] * d[i] for i in range(len(P)))
        r2 = sum(sum(t[i] * d[i] for i in range(len(P))) ** 2 for t in bd["t"])
        env = np.exp(-r2 / (2 * bd["sig"] ** 2)) * amp_scale
        A = bd["ratio"] * bd["w"]
        mag = -A * np.tanh(s / bd["w"]) * env
        return [mag * bd["n"][i] for i in range(self.ks)], s, env

    # evaluation -------------------------------------------------------------
    def eval(self, P: list[np.ndarray], t: float | np.ndarray = 0.0, noise_idx=()):
        ks = self.ks
        t = np.asarray(t, dtype=np.float64)
        u = [np.full(P[0].shape, self.shift[i]) + 0.0 * t for i in range(ks)]
        for i in range(ks):
            for j in range(ks):
                u[i] = u[i] + self.lin[i, j] * (P[j] - self.extent[j] / 2)
        for k, amp, ph, om in self.modes:
            arg = sum(k[i] * P[i] for i in range(ks)) + ph + om * t
            s = np.sin(arg)
            for i in range(ks):
                u[i] = u[i] + amp[i] * s
        for c, sig, a in self.bumps:
            r2 = sum((P[i] - c[i]) ** 2 for i in range(ks))
            g = np.exp(-r2 / (2 * sig**2))
            for i in range(ks):
                u[i] = u[i] + a[i] * g
        k, amp = self.ripple
        rip = amp * np.sin(sum(k[i] * P[i] for i in range(ks)))
        for i in range(ks):
            u[i] = u[i] + rip * (1.0 if i == 0 else 0.5)
        # discontinuities; cracks drift and breathe slowly with t
        drift = self.h * 1.5 * float(t)  # crack/band centres move 1.5 samples per unit t
        breathe = 1.0 + 0.3 * np.sin(0.4 * t)
        lab = np.zeros(P[0].shape, dtype=np.uint8)
        for direction, cr, code in self.cracks:
            du = self._crack(P, cr, direction, drift, breathe)
            u = [u[i] + du[i] for i in range(ks)]
            lab = np.where(self._crack_label(P, cr, drift), np.uint8(code), lab)
        for bd, code in ((self.band, 3), (self.fold, 4)):
            du, s, env = self._profile(P, bd, drift * 0, 1.0)
            u = [u[i] + du[i] for i in range(ks)]
            if code == 3:
                m = (np.abs(s) < 2 * bd["w"]) & (env > 0.3)
            else:
                m = bd["ratio"] * env / np.cosh(s / bd["w"]) ** 2 > 1.0
            lab = np.where(m, np.uint8(code), lab)
        if noise_idx:
            for i in range(ks):
                u[i] = u[i] + self.noise * _hash_noise(
                    *noise_idx, np.full(1, i), seed=self.seed
                )
        return np.stack(u, axis=-1), lab


# -- presets -------------------------------------------------------------------
@dataclass(frozen=True)
class Preset:
    name: str
    description: str
    shape: tuple[int, ...]  # array shape, component axis last
    chunks: tuple[int, ...]
    axis_roles: tuple[str, ...]
    spacing: float = 4.0  # image voxels per field sample (spatial axes)
    seed: int = 0
    misaligned: tuple[int, ...] = ()  # stack index of misaligned sections/timepoints
    crack_range: tuple[int, int] | None = None  # 2d_stack: sections that carry the cracks

    @property
    def spatial_axes(self) -> tuple[int, ...]:
        return tuple(i for i, r in enumerate(self.axis_roles) if r == "spatial")

    @property
    def stack_axis(self) -> int | None:
        for i, r in enumerate(self.axis_roles):
            if r in ("smooth", "independent"):
                return i
        return None

    @property
    def ks(self) -> int:
        return len(self.spatial_axes)

    def _fields(self) -> dict:
        extent = np.array([self.shape[a] for a in self.spatial_axes], dtype=np.float64)
        extent = extent * self.spacing
        hs = np.full(self.ks, self.spacing)
        return dict(extent=extent, spacing=hs)

    def region(self, sel: tuple[slice, ...] | None = None):
        """Return (u, labels) for a region given as slices over the non-component axes."""
        nd = len(self.shape) - 1
        if sel is None:
            sel = tuple(slice(0, n) for n in self.shape[:-1])
        sel = tuple(slice(*s.indices(self.shape[i])) for i, s in enumerate(sel))
        grids = np.meshgrid(*[np.arange(s.start, s.stop) for s in sel], indexing="ij")
        P = [grids[a] * self.spacing for a in self.spatial_axes]
        fk = self._fields()
        sa = self.stack_axis
        if sa is None:
            f = Field(self.ks, seed=self.seed, **fk)
            u, lab = f.eval(P, 0.0, noise_idx=tuple(grids))
        elif self.axis_roles[sa] == "independent":
            u = np.zeros((*grids[0].shape, self.ks))
            lab = np.zeros(grids[0].shape, dtype=np.uint8)
            tv = np.arange(sel[sa].start, sel[sa].stop)
            for k, t in enumerate(tv):
                sl = [slice(None)] * nd
                sl[sa] = k
                sl = tuple(sl)
                f = Field(self.ks, seed=self.seed + 1000 * int(t), **fk)
                u[sl], lab[sl] = f.eval(
                    [p[sl] for p in P], 0.0, noise_idx=tuple(g[sl] for g in grids)
                )
        else:  # smooth stack / time axis
            f = Field(self.ks, seed=self.seed, **fk)
            tv = np.arange(sel[sa].start, sel[sa].stop)
            u = np.zeros((*grids[0].shape, self.ks))
            lab = np.zeros(grids[0].shape, dtype=np.uint8)
            for k, t in enumerate(tv):
                sl = [slice(None)] * nd
                sl[sa] = k
                sl = tuple(sl)
                uu, ll = f.eval(
                    [p[sl] for p in P], float(t) * 0.5, noise_idx=tuple(g[sl] for g in grids)
                )
                if self.crack_range is not None and not (
                    self.crack_range[0] <= t < self.crack_range[1]
                ):
                    # remove the cracks outside the range: evaluate without them
                    uu, ll = _without_cracks(
                        f, [p[sl] for p in P], float(t) * 0.5, tuple(g[sl] for g in grids)
                    )
                u[sl], lab[sl] = uu, ll
        if sa is not None and self.misaligned:
            tv = np.arange(sel[sa].start, sel[sa].stop)
            off = np.array([7.0, -5.0, 4.0][: self.ks])
            for k, t in enumerate(tv):
                if t in self.misaligned:
                    sl = [slice(None)] * nd
                    sl[sa] = k
                    u[tuple(sl)] += off
                    lab[tuple(sl)] = 5
        return u.astype(np.float32), lab

    def full(self):
        return self.region(None)


def _without_cracks(f: Field, P, t, noise_idx):
    cracks = f.cracks
    f.cracks = []
    try:
        u, lab = f.eval(P, t, noise_idx=noise_idx)
    finally:
        f.cracks = cracks
    return u, lab


S2 = ("spatial", "spatial", "component")
S3 = ("spatial", "spatial", "spatial", "component")

_FULL = {
    "2d": Preset("2d", "2D section alignment field", (384, 384, 2), (100, 100, 2), S2),
    "2d_stack": Preset(
        "2d_stack",
        "2D section fields stacked along z (smooth across sections)",
        (24, 256, 256, 2),
        (8, 100, 100, 2),
        ("smooth", *S2),
        misaligned=(5, 17),
        crack_range=(8, 16),
    ),
    "3d": Preset("3d", "3D volume field", (160, 160, 160, 3), (64, 64, 64, 3), S3),
    "3d_time_stack": Preset(
        "3d_time_stack",
        "3D fields stacked over time, independent per timepoint",
        (4, 96, 96, 96, 3),
        (2, 64, 64, 64, 3),
        ("independent", *S3),
    ),
    "4d": Preset(
        "4d",
        "4D field, smooth across time, with a moving crack",
        (8, 96, 96, 96, 3),
        (4, 64, 64, 64, 3),
        ("smooth", *S3),
        misaligned=(6,),
    ),
}

_SMALL = {
    "2d": (130, 130, 2),
    "2d_stack": (10, 80, 80, 2),
    "3d": (56, 56, 56, 3),
    "3d_time_stack": (3, 40, 40, 40, 3),
    "4d": (5, 40, 40, 40, 3),
}
_SMALL_CHUNKS = {
    "2d": (48, 48, 2),
    "2d_stack": (4, 48, 48, 2),
    "3d": (24, 24, 24, 3),
    "3d_time_stack": (2, 24, 24, 24, 3),
    "4d": (3, 24, 24, 24, 3),
}

PRESET_NAMES = tuple(_FULL)


def preset(name: str, scale: str = "full") -> Preset:
    p = _FULL[name]
    if scale == "full":
        return p
    if scale == "small":
        from dataclasses import replace

        mis = tuple(m for m in p.misaligned if m < _SMALL[name][0]) or (
            (2,) if p.misaligned else ()
        )
        cr = (3, 7) if p.crack_range else None
        return replace(
            p, shape=_SMALL[name], chunks=_SMALL_CHUNKS[name], misaligned=mis, crack_range=cr
        )
    raise ValueError(scale)


# -- a procedural test image ----------------------------------------------------
def test_image(P: list[np.ndarray], period: float = 32.0) -> np.ndarray:
    """Analytic image in [0, 1] at continuous image coordinates (any dimension).

    Dark grid lines every `period` voxels on the last two axes, plus
    band-limited texture at proportional scales, so both geometry and fine
    detail show distortion.
    """
    y, x = P[-2], P[-1]
    z = P[0] if len(P) > 2 else 0.0 * x
    f = period / 32.0
    tex = (
        0.5
        + 0.12 * np.sin(2 * np.pi * x / (13.0 * f)) * np.sin(2 * np.pi * y / (17.0 * f))
        + 0.10 * np.sin(2 * np.pi * (x + 0.7 * y + 0.3 * z) / (41.0 * f))
        + 0.08 * np.cos(2 * np.pi * (0.6 * x - y + 0.5 * z) / (29.0 * f))
    )

    width = 1.6 * max(f, 1.0) ** 0.5

    def line(c, per):
        d = np.abs(((c + per / 2) % per) - per / 2)
        return np.exp(-((d / width) ** 2))

    g = np.maximum(line(x, period), line(y, period))
    if len(P) > 2:
        g = np.maximum(g, 0.5 * line(z, 2 * period))
    return np.clip(tex * (1 - 0.75 * g), 0, 1)
