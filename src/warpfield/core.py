"""Chunk encoder and decoder (numpy + zstandard only).

Pipeline for one chunk:

1. Reorder to the canonical array of shape (B, *P): the `independent` axes,
   then the `component` axis, flattened into one batch axis B; then the
   predicted (`spatial` and `smooth`) axes P in their original order. Batch
   rows are coded independently but in one vectorized pass.
2. Give every sample an error class: 0 interior (`error_bound`), 1 face
   (`face_error_bound`, any predicted index at 0 or N-1), 2 tight
   (`jacobian.tight_error_bound`), 3 verbatim (stored exactly). With a ladder
   of tight bounds (`tight_error_bound` a list, loosest first), tier t > 0 is
   class 3 + t.
3. Run the closed-loop multilevel interpolation pass (see `predict`). For each
   sample: q = rint((x - pred) / (2 eps)); recon = dtype(pred + q * 2 eps). If
   |x - recon| > eps (float rounding, huge residuals, non-finite input) the
   sample becomes verbatim on the spot, so every sample satisfies its bound by
   construction.
4. If `jacobian` is configured, the pass checks det(I + du/dx) of every cell
   as soon as a block codes its last corner. If the cell newly falls below
   `min_det`, or its determinant changes sign, its corners in that block
   (nothing has been predicted from them yet) move to the next tight tier,
   then to verbatim, and are re-quantized on the spot. Cells that this cannot
   fix (their earlier corners are to blame) have all their corners promoted
   after the pass, and the pass is rerun. Verbatim cells reproduce the
   original determinant exactly, so the loop terminates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from warpfield import _backend
from warpfield import format as fmt
from warpfield import jacobian as jac
from warpfield.predict import (
    BOUNDARIES,
    PREDICTORS,
    known,
    new_points,
    predict,
    stencil,
    top_stride,
)

ROLES = ("spatial", "smooth", "independent", "component")
PREDICTED = ("spatial", "smooth")
INTERIOR, FACE, TIGHT, VERBATIM = 0, 1, 2, 3
MAX_TIGHT_ROUNDS = 3
MAX_TIERS = 4


def tight_class(tier: int) -> int:
    """Class of tight tier `tier` (0 is TIGHT; later tiers follow VERBATIM)."""
    return TIGHT if tier == 0 else VERBATIM + tier


def class_tier(c: np.ndarray) -> np.ndarray:
    """Tight tier of tight classes (as `tight_class` inverted)."""
    c = np.asarray(c)
    return np.where(c == TIGHT, 0, c.astype(np.int64) - VERBATIM).astype(np.uint8)


def is_tight(c: np.ndarray) -> np.ndarray:
    return (c == TIGHT) | (c > VERBATIM)


class WarpFieldSpecError(ValueError):
    pass


@dataclass(frozen=True)
class JacobianConfig:
    grid_spacing: tuple[float, ...]
    min_det: float = 0.0
    # one bound, or a ladder of decreasing bounds (tiers), loosest first
    tight_error_bound: float | tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        t = self.tight_error_bound
        if isinstance(t, (list, tuple)):
            t = tuple(float(v) for v in t)
            object.__setattr__(self, "tight_error_bound", t[0] if len(t) == 1 else t)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"grid_spacing": list(self.grid_spacing), "min_det": self.min_det}
        t = self.tight_error_bound
        if t is not None:
            d["tight_error_bound"] = list(t) if isinstance(t, tuple) else t
        return d


@dataclass(frozen=True)
class Config:
    axis_roles: tuple[str, ...]
    error_bound: float = 0.25
    face_error_bound: float | None = None
    predictor: str = "eno"
    boundary: str = "clamped"
    jacobian: JacobianConfig | None = None

    def __post_init__(self) -> None:
        roles = tuple(self.axis_roles)
        object.__setattr__(self, "axis_roles", roles)
        for r in roles:
            if r not in ROLES:
                raise WarpFieldSpecError(f"unknown axis role {r!r}; expected one of {ROLES}")
        if roles.count("component") > 1:
            raise WarpFieldSpecError("at most one axis may have role 'component'")
        if not any(r in PREDICTED for r in roles):
            raise WarpFieldSpecError("at least one axis must be 'spatial' or 'smooth'")
        if not (self.error_bound > 0 and np.isfinite(self.error_bound)):
            raise WarpFieldSpecError("error_bound must be a positive finite number")
        if self.face_error_bound is not None and not (
            0 < self.face_error_bound <= self.error_bound
        ):
            raise WarpFieldSpecError("face_error_bound must be in (0, error_bound]")
        if self.predictor not in PREDICTORS:
            raise WarpFieldSpecError(f"predictor must be one of {PREDICTORS}")
        if self.boundary not in BOUNDARIES:
            raise WarpFieldSpecError(f"boundary must be one of {BOUNDARIES}")
        j = self.jacobian
        if isinstance(j, dict):
            j = JacobianConfig(
                grid_spacing=tuple(float(s) for s in j["grid_spacing"]),
                min_det=float(j.get("min_det", 0.0)),
                tight_error_bound=j.get("tight_error_bound"),
            )
            object.__setattr__(self, "jacobian", j)
        if j is not None:
            ns = roles.count("spatial")
            if "component" not in roles:
                raise WarpFieldSpecError("jacobian needs a 'component' axis")
            if not 1 <= ns <= 3:
                raise WarpFieldSpecError("jacobian supports 1 to 3 spatial axes")
            if len(j.grid_spacing) != ns:
                raise WarpFieldSpecError(
                    f"jacobian.grid_spacing needs {ns} entries, one per spatial axis"
                )
            if any(s <= 0 for s in j.grid_spacing):
                raise WarpFieldSpecError("jacobian.grid_spacing entries must be positive")
            t = j.tight_error_bound
            ts = t if isinstance(t, tuple) else (t,) if t is not None else ()
            if any(not 0 < v <= self.error_bound for v in ts):
                raise WarpFieldSpecError("tight_error_bound must be in (0, error_bound]")
            if isinstance(t, tuple) and (
                not 1 <= len(t) <= MAX_TIERS
                or any(a <= b for a, b in zip(t, t[1:], strict=False))
            ):
                raise WarpFieldSpecError(
                    f"a tight_error_bound ladder needs 1 to {MAX_TIERS} strictly "
                    "decreasing bounds"
                )

    # -- derived ------------------------------------------------------------
    @property
    def eps_face(self) -> float:
        return self.error_bound / 4 if self.face_error_bound is None else self.face_error_bound

    @property
    def tight_bounds(self) -> tuple[float, ...]:
        """The tight tiers' bounds, loosest first."""
        j = self.jacobian
        t = None if j is None else j.tight_error_bound
        if t is None:
            return (self.error_bound / 8,)
        return t if isinstance(t, tuple) else (float(t),)

    @property
    def eps_tight(self) -> float:
        return self.tight_bounds[0]

    @property
    def tight_classes(self) -> tuple[int, ...]:
        return tuple(tight_class(t) for t in range(len(self.tight_bounds)))

    def eps_table(self) -> np.ndarray:
        """Bound per class: interior, face, tight, verbatim, then tiers 1, 2, ..."""
        t = self.tight_bounds
        return np.array([self.error_bound, self.eps_face, t[0], 0.0, *t[1:]], dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "axis_roles": list(self.axis_roles),
            "error_bound": self.error_bound,
            "face_error_bound": self.eps_face,
            "predictor": self.predictor,
        }
        if self.boundary != "clamped":
            d["boundary"] = self.boundary
        if self.jacobian is not None:
            d["jacobian"] = self.jacobian.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Config:
        return cls(**d)


# -- canonical layout ---------------------------------------------------------
@dataclass(frozen=True)
class Layout:
    perm: tuple[int, ...]
    shape: tuple[int, ...]  # input chunk shape
    n_indep: int  # product of independent extents
    n_comp: int
    pred_shape: tuple[int, ...]
    pred_roles: tuple[str, ...]

    @property
    def canonical_shape(self) -> tuple[int, ...]:
        return (self.n_indep * self.n_comp, *self.pred_shape)

    @property
    def spatial_in_pred(self) -> tuple[int, ...]:
        return tuple(i for i, r in enumerate(self.pred_roles) if r == "spatial")


def layout(roles: tuple[str, ...], shape: tuple[int, ...]) -> Layout:
    if len(roles) != len(shape):
        raise WarpFieldSpecError(
            f"chunk rank {len(shape)} does not match {len(roles)} axis_roles"
        )
    ind = [i for i, r in enumerate(roles) if r == "independent"]
    comp = [i for i, r in enumerate(roles) if r == "component"]
    pred = [i for i, r in enumerate(roles) if r in PREDICTED]
    perm = tuple(ind + comp + pred)
    n_ind = int(np.prod([shape[i] for i in ind])) if ind else 1
    n_comp = shape[comp[0]] if comp else 1
    return Layout(
        perm,
        tuple(shape),
        n_ind,
        n_comp,
        tuple(shape[i] for i in pred),
        tuple(roles[i] for i in pred),
    )


def to_canonical(arr: np.ndarray, lay: Layout) -> np.ndarray:
    return np.transpose(arr, lay.perm).reshape(lay.canonical_shape)


def from_canonical(c: np.ndarray, lay: Layout) -> np.ndarray:
    t_shape = tuple(lay.shape[i] for i in lay.perm)
    return np.transpose(c.reshape(t_shape), np.argsort(lay.perm))


def face_classes(lay: Layout) -> np.ndarray:
    cls = np.zeros(lay.canonical_shape, dtype=np.uint8)
    for j, n in enumerate(lay.pred_shape):
        if n < 2:
            continue
        sl = [slice(None)] * cls.ndim
        for end in (0, n - 1):
            sl[j + 1] = end
            cls[tuple(sl)] = FACE
    return cls


# -- the closed-loop pass -----------------------------------------------------
@dataclass
class _PassResult:
    recon: np.ndarray
    codes: np.ndarray


def level_eps(base: np.ndarray, h: int, schedule: tuple[float, float]) -> np.ndarray:
    """Level schedule (QoZ / HPEZ): the level with interpolation step h = 2^L
    is coded at eps / min(alpha^L, beta). alpha^L by repeated multiplication,
    as in the Rust crate, so both give the same bits."""
    alpha, beta = schedule
    f, s = 1.0, 1
    while s < h:
        f *= alpha
        s *= 2
    return base / min(f, beta)


class _InPass:
    """Jacobian check during the pass: which cells each block completes, and
    the samples each cell's determinant depends on (x and x + e_b)."""

    def __init__(self, cfg: Config, lay: Layout, d0: np.ndarray) -> None:
        self.j, self.lay, self.d0 = cfg.jacobian, lay, d0
        self.spatial = tuple(1 + p for p in lay.spatial_in_pred)  # in (I, *P)
        self.coded = np.zeros(lay.pred_shape, dtype=bool)
        self.done = np.zeros(d0.shape[1:], dtype=bool)  # cell shape without I
        eps = cfg.eps_table()
        ladder = cfg.tight_classes
        # next class per class: the loosest tier tighter than it, else verbatim
        self.nxt = np.full(eps.size, VERBATIM, dtype=np.uint8)
        for c in range(eps.size):
            for t in reversed(ladder):
                if eps[t] < eps[c]:
                    self.nxt[c] = t

    def next_class(self, c: np.ndarray) -> np.ndarray:
        return self.nxt[c]

    def _shifted(self, a: np.ndarray, b: int) -> np.ndarray:
        sl = [slice(None)] * a.ndim
        ax = self.spatial[b] - 1
        sl[ax] = slice(1, None)
        return a[tuple(sl)]

    def complete(self, lists: list[np.ndarray]) -> np.ndarray:
        """Mark the block's positions coded; return (I, *cells) newly complete."""
        self.coded[np.ix_(*lists)] = True
        cut = [slice(None)] * self.coded.ndim
        for b in range(len(self.spatial)):
            cut[self.spatial[b] - 1] = slice(0, -1)
        all_in = self.coded[tuple(cut)].copy()
        for b in range(len(self.spatial)):
            sh = self._shifted(self.coded, b)
            sl = list(cut)
            sl[self.spatial[b] - 1] = slice(None)
            all_in &= sh[tuple(sl)]
        new = all_in & ~self.done
        self.done |= new
        return np.broadcast_to(new, self.d0.shape)

    def corners(self, cells: np.ndarray) -> np.ndarray:
        """Canonical mask of the samples (all components) the cells depend on."""
        lay = self.lay
        m = np.zeros((lay.n_indep, *lay.pred_shape), dtype=bool)
        base = tuple(slice(0, n) for n in cells.shape)
        m[base] |= cells
        for ax in self.spatial:
            sl = list(base)
            sl[ax] = slice(1, cells.shape[ax] + 1)
            m[tuple(sl)] |= cells
        m = np.broadcast_to(m[:, None], (lay.n_indep, lay.n_comp, *lay.pred_shape))
        return m.reshape(lay.canonical_shape)


def _run_pass(
    cfg: Config,
    cls: np.ndarray,
    dtype: np.dtype,
    x: np.ndarray | None = None,
    codes: np.ndarray | None = None,
    recon_init: np.ndarray | None = None,
    schedule: tuple[float, float] = (1.0, 1.0),
    check: _InPass | None = None,
) -> _PassResult:
    """Encode (x given) or decode (codes given) one canonical chunk.

    In encode mode `cls` is updated in place: samples that fail their bound
    become VERBATIM, and with `check` (a Jacobian constraint) the corners of
    cells that a block completes and breaks are tightened on the spot.
    """
    encoding = x is not None
    shape = cls.shape
    eps_base = cfg.eps_table()
    eps = eps_base
    R = np.zeros(shape, dtype=np.float64) if recon_init is None else recon_init
    out_codes: list[np.ndarray] = []
    cursor = 0
    batch = np.arange(shape[0], dtype=np.int64)
    pshape = shape[1:]
    boundary = cfg.boundary

    def quantize(xb, pred, c):
        e = eps[c]
        w = 2.0 * e
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            q = np.where(c != VERBATIM, np.rint((xb - pred) / w), 0.0)
            r = (pred + q * w).astype(dtype).astype(np.float64)
            ok = (c != VERBATIM) & np.isfinite(xb) & (np.abs(xb - r) <= e)
            ok &= np.abs(q) < 2.0**52
        # samples that fail their bound are stored exactly
        return q, np.where(ok, r, xb), ok, np.where(ok, c, VERBATIM).astype(np.uint8)

    def block(lists: list[np.ndarray], pred: np.ndarray) -> None:
        nonlocal cursor
        index = np.ix_(batch, *lists)
        c = cls[index]
        pred = np.where(np.isfinite(pred), pred, 0.0)
        if encoding:
            xb = x[index]
            q, r, ok, c = quantize(xb, pred, c)
            R[index] = r
            cls[index] = c
            if check is not None:
                # cells this block completes: a broken one gets its corners in
                # this block (which nothing has been predicted from yet) one
                # tier tighter, then exact, until it holds or they are exact
                cells = check.complete(lists)
                while cells.any():
                    # every cell the block completes is re-checked: tightening a
                    # shared corner can break a neighbour
                    bad = cells & jac.violations(
                        check.d0, _det(R, check.lay, check.j), check.j.min_det
                    )
                    hit = check.corners(bad)[index] & (c != VERBATIM)
                    if not hit.any():
                        break
                    c = np.where(hit, check.next_class(c), c).astype(np.uint8)
                    q, r, ok, c = quantize(xb, pred, c)
                    R[index] = r
                    cls[index] = c
            out_codes.append(q[ok].astype(np.int64))
        else:
            e = eps[c]
            w = 2.0 * e
            nv = c != VERBATIM
            n = int(nv.sum())
            qb = codes[cursor : cursor + n].astype(np.float64)
            cursor += n
            r = R[index]
            r[nv] = (pred[nv] + qb * w[nv]).astype(dtype).astype(np.float64)
            R[index] = r

    # anchors: the chunk corners (or the coarsest lattice), predicted as zero
    S = top_stride(pshape)
    lists = [known(S, n, boundary) for n in pshape]
    eps = level_eps(eps_base, S, schedule)
    block(lists, np.zeros((shape[0], *(len(a) for a in lists))))

    h = S // 2
    while h >= 1:
        tau = cfg.error_bound / float(h * h)
        eps = level_eps(eps_base, h, schedule)
        for j, n in enumerate(pshape):
            tg = new_points(h, n, boundary)
            if tg.size == 0:
                continue
            st = stencil(tg, known(2 * h, n, boundary))
            other = [
                known(h, pshape[i], boundary) if i < j else known(2 * h, pshape[i], boundary)
                for i in range(len(pshape))
            ]
            vals = []
            for slot in range(4):
                lst = list(other)
                lst[j] = st.slots[slot]
                vals.append(R[np.ix_(batch, *lst)])
            pred = predict(st, vals, j + 1, len(shape), cfg.predictor, tau)
            lst = list(other)
            lst[j] = tg
            block(lst, pred)
        h //= 2

    codes_out = np.concatenate(out_codes) if out_codes else np.zeros(0, dtype=np.int64)
    if not encoding and cursor != codes.size:
        raise fmt.WarpFieldFormatError(f"{codes.size - cursor} unused quantization codes")
    return _PassResult(R, codes_out)


# -- Jacobian helpers on the canonical layout -----------------------------------
def _det(c: np.ndarray, lay: Layout, j: JacobianConfig) -> np.ndarray:
    u = c.reshape(lay.n_indep, lay.n_comp, *lay.pred_shape)
    spatial = tuple(2 + p for p in lay.spatial_in_pred)
    return jac.det_jacobian(u, 1, spatial, j.grid_spacing)


def _cells_to_samples(cells: np.ndarray, lay: Layout) -> np.ndarray:
    spatial = tuple(1 + p for p in lay.spatial_in_pred)
    m = jac.cell_samples(cells, spatial)  # (I, *P)
    m = np.broadcast_to(m[:, None], (lay.n_indep, lay.n_comp, *lay.pred_shape))
    return m.reshape(lay.canonical_shape)


# -- public API -----------------------------------------------------------------
@dataclass
class EncodeStats:
    n_samples: int
    n_codes: int
    n_tight: int
    n_verbatim: int
    jacobian_rounds: int
    nbytes: int


def _config_json(cfg: Config) -> str:
    return json.dumps(cfg.to_dict())


def encode(
    arr: np.ndarray,
    cfg: Config,
    *,
    stats: bool = False,
    backend: str | None = None,
    schedule: tuple[float, float] | None = None,
    min_classes: np.ndarray | None = None,
    effort: str = "fast",
    zstd_level: int | None = None,
):
    """Encode one chunk. Returns bytes, or (bytes, EncodeStats, classes) if `stats`.

    `effort` is "fast" (the default: tune on the chunk's central 33^d, choose
    the code coder at zstd level 3, compress at level 12) or "max" (tune on
    65^d, compress both coders at level 19): about 2-7x slower, 1-17% smaller.
    Decoding and every guarantee are the same either way. `zstd_level`
    overrides the effort's level.

    `backend` is "rust", "python" or "auto"/None (Rust when installed); see `_backend`.
    `schedule` is the level schedule (alpha, beta), both >= 1; None tunes it per
    chunk (`tune`), (1, 1) is the uniform schedule of format version 1.
    `min_classes` (uint8, the chunk's shape) forces samples to a tight tier
    (`tight_class`) or VERBATIM from the start, where that tightens them;
    writers use it for cells straddling chunk faces (see `warpfield.halo`).
    """
    if effort not in EFFORTS:
        raise WarpFieldSpecError(f"effort must be one of {tuple(EFFORTS)}, not {effort!r}")
    if min_classes is not None:
        min_classes = np.ascontiguousarray(min_classes, dtype=np.uint8)
        if min_classes.shape != np.shape(arr):
            raise WarpFieldSpecError(
                f"min_classes has shape {min_classes.shape}, the chunk {np.shape(arr)}"
            )
    if schedule is not None:
        schedule = (float(schedule[0]), float(schedule[1]))
        if not all(np.isfinite(v) and v >= 1 for v in schedule):
            raise WarpFieldSpecError("schedule alpha and beta must be finite and >= 1")
    arr = np.asarray(arr)
    dtype = arr.dtype
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise WarpFieldSpecError(f"data type must be float32 or float64, not {dtype}")
    if _backend.resolve(backend) == "rust":
        lay = layout(cfg.axis_roles, arr.shape)  # same spec errors as the reference
        if cfg.jacobian is not None and lay.n_comp != len(lay.spatial_in_pred):
            raise WarpFieldSpecError(
                "jacobian needs one component per spatial axis "
                f"({len(lay.spatial_in_pred)}), got {lay.n_comp}"
            )
        try:
            buf, cls, n_codes, n_tight, n_verb, rounds = _backend.rs.encode(
                np.ascontiguousarray(arr),
                _config_json(cfg),
                classes=stats,
                schedule=schedule,
                min_classes=min_classes,
                effort=effort,
                zstd_level=zstd_level,
            )
        except _backend.rs.SpecError as e:
            raise WarpFieldSpecError(str(e)) from None
        if not stats:
            return buf
        return buf, EncodeStats(arr.size, n_codes, n_tight, n_verb, rounds, len(buf)), cls
    return _encode_py(
        arr,
        cfg,
        stats=stats,
        schedule=schedule,
        min_classes=min_classes,
        effort=effort,
        zstd_level=zstd_level,
    )


def _code_canonical(cfg: Config, lay: Layout, x: np.ndarray, dtype, schedule, min_classes=None):
    """The closed loop on one canonical array: code it with `schedule`, then,
    with a Jacobian constraint, tighten or store verbatim the samples of
    violating cells until none are left. Returns (classes, pass result, rounds)."""
    base = face_classes(lay)
    eps = cfg.eps_table()
    ladder = cfg.tight_classes
    # forced class per sample, 0 = none (the face class applies)
    force = np.zeros(base.shape, dtype=np.uint8)
    if min_classes is not None:
        m = np.asarray(min_classes, dtype=np.uint8)
        if int(m.max(initial=0)) >= eps.size:
            raise WarpFieldSpecError("min_classes names a tight tier the configuration lacks")
        # only where it tightens (a face bound may already be below a tier)
        sel = (m >= TIGHT) & (eps[m] < eps[base])
        force[sel] = m[sel]
    j = cfg.jacobian
    d0 = _det(x, lay, j) if j is not None else None
    max_rounds = MAX_TIGHT_ROUNDS + len(ladder) - 1
    rounds = 0
    while True:
        cur = np.where(force != 0, force, base)
        cls = cur.copy()
        check = _InPass(cfg, lay, d0) if j is not None else None
        res = _run_pass(cfg, cls, dtype, x=x, schedule=schedule, check=check)
        if j is None:
            return cls, res, rounds
        bad_cells = jac.violations(d0, _det(res.recon, lay, j), j.min_det)
        if not bad_cells.any():
            return cls, res, rounds
        rounds += 1
        hit = _cells_to_samples(bad_cells, lay) & (force != VERBATIM)
        if not hit.any():  # cannot happen: verbatim cells reproduce d0 exactly
            raise RuntimeError("jacobian promotion made no progress")
        # the loosest tier tighter than the sample's class, else verbatim
        nxt = np.full(base.shape, VERBATIM, dtype=np.uint8)
        if rounds <= max_rounds:
            ecur = eps[cur]
            for c in reversed(ladder):
                nxt[eps[c] < ecur] = c
        force[hit] = nxt[hit]


# Per-chunk level schedule, after QoZ / HPEZ: code the central TUNE_SAMPLE^d
# sub-block with each candidate and keep the cheapest. Same candidates, cost and
# tie-break as the Rust crate (`tune.rs`), so both pick the same schedule.
TUNE_CANDIDATES = (
    (1.0, 1.0),
    (1.1, 1.5),
    (1.1, 3.0),
    (1.25, 2.0),
    (1.5, 2.0),
    (1.5, 3.0),
    (2.0, 4.0),
)
TUNE_SAMPLE = 65  # effort "max"; "fast" tunes on 33^d

# Encoder effort: tuning sub-block extent, zstd level, coder-choice level.
# Same settings as the Rust crate's `Effort`.
EFFORTS = {"fast": (33, 12, 3), "max": (TUNE_SAMPLE, 19, None)}


def tune_cost(codes: np.ndarray, n_verbatim: int) -> int:
    """Σ over nonzero codes of 1 + bit length of zigzag(q), plus 64 per verbatim sample."""
    z = fmt.zigzag(codes[codes != 0])
    bitlen = np.zeros(z.size, dtype=np.int64)
    for b in range(64):
        bitlen += z >= (np.uint64(1) << np.uint64(b))
    return int(64 * n_verbatim + z.size + bitlen.sum())


def tune(
    cfg: Config, lay: Layout, x: np.ndarray, dtype, extent: int = TUNE_SAMPLE
) -> tuple[float, float]:
    sl = [slice(None)]
    sub = []
    for n in lay.pred_shape:
        if n > extent:
            s0 = (n - extent) // 2
            sl.append(slice(s0, s0 + extent))
            sub.append(extent)
        else:
            sl.append(slice(None))
            sub.append(n)
    xs = np.ascontiguousarray(x[tuple(sl)])
    sublay = replace(lay, pred_shape=tuple(sub))
    best = None
    for cand in TUNE_CANDIDATES:
        cls, res, _ = _code_canonical(cfg, sublay, xs, dtype, cand)
        c = tune_cost(res.codes, int((cls == VERBATIM).sum()))
        if best is None or c < best[0]:
            best = (c, cand)
    return best[1]


def _encode_py(
    arr: np.ndarray,
    cfg: Config,
    *,
    stats: bool = False,
    schedule=None,
    min_classes=None,
    effort: str = "fast",
    zstd_level: int | None = None,
):
    dtype = arr.dtype
    lay = layout(cfg.axis_roles, arr.shape)
    x = to_canonical(arr, lay).astype(np.float64)
    extent, level, pick = EFFORTS[effort]
    if zstd_level is not None:
        level = zstd_level
    if schedule is None:
        schedule = tune(cfg, lay, x, dtype, extent)
    mc = to_canonical(min_classes, lay) if min_classes is not None else None
    cls, res, rounds = _code_canonical(cfg, lay, x, dtype, tuple(schedule), mc)

    verb = np.flatnonzero(cls.ravel() == VERBATIM)
    tight = np.flatnonzero(is_tight(cls.ravel()))
    tiers = class_tier(cls.ravel()[tight])
    vals = x.ravel()[verb].astype(dtype)
    buf = fmt.write_chunk(
        lay.canonical_shape, res.codes, tight, verb, vals, schedule, tiers, level, pick
    )
    if not stats:
        return buf
    st = EncodeStats(arr.size, res.codes.size, tight.size, verb.size, rounds, len(buf))
    return buf, st, from_canonical(cls, lay)


def decode(
    buf: bytes, shape: tuple[int, ...], dtype: Any, cfg: Config, *, backend: str | None = None
) -> np.ndarray:
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise WarpFieldSpecError(f"data type must be float32 or float64, not {dtype}")
    if _backend.resolve(backend) == "rust":
        layout(cfg.axis_roles, tuple(shape))
        try:
            return _backend.rs.decode(bytes(buf), list(shape), dtype.name, _config_json(cfg))
        except _backend.rs.FormatError as e:
            raise fmt.WarpFieldFormatError(str(e)) from None
        except _backend.rs.SpecError as e:
            raise WarpFieldSpecError(str(e)) from None
    return _decode_py(buf, shape, dtype, cfg)


def _decode_py(buf: bytes, shape: tuple[int, ...], dtype: Any, cfg: Config) -> np.ndarray:
    dtype = np.dtype(dtype)
    lay = layout(cfg.axis_roles, tuple(shape))
    cshape, codes, tight, verb, vals, tiers = fmt.read_chunk(bytes(buf), dtype, tiers=True)
    schedule = fmt.read_header(bytes(buf))[3]
    if cshape != lay.canonical_shape:
        raise fmt.WarpFieldFormatError(
            f"chunk was encoded with canonical shape {cshape}, expected {lay.canonical_shape}"
        )
    cls = face_classes(lay)
    if tiers.size and int(tiers.max()) >= len(cfg.tight_bounds):
        raise fmt.WarpFieldFormatError("chunk uses a tight tier the configuration lacks")
    cls.flat[tight] = np.where(tiers == 0, TIGHT, VERBATIM + tiers.astype(np.int64))
    cls.flat[verb] = VERBATIM
    R = np.zeros(cls.shape, dtype=np.float64)
    R.flat[verb] = vals.astype(np.float64)
    res = _run_pass(cfg, cls, dtype, codes=codes, recon_init=R, schedule=schedule)
    return from_canonical(res.recon, lay).astype(dtype)


def error_bounds(shape: tuple[int, ...], cfg: Config, classes: np.ndarray | None = None):
    """Per-sample guaranteed absolute error bound for a chunk of `shape`."""
    lay = layout(cfg.axis_roles, tuple(shape))
    cls = to_canonical(classes, lay) if classes is not None else face_classes(lay)
    return from_canonical(cfg.eps_table()[cls], lay)
