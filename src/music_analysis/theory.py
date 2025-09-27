from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

N_PCS_DEFAULT = 12


def pcs_from_mask(mask: int, n: int = N_PCS_DEFAULT) -> List[int]:
    return [i for i in range(n) if (mask >> i) & 1]


def mask_from_pcs(pcs: Sequence[int]) -> int:
    m = 0
    for p in pcs:
        m |= 1 << (p % 64)
    return m


def step_vector_from_pcs(pcs: Sequence[int], n: int = N_PCS_DEFAULT) -> List[int]:
    pcs = sorted(pcs)
    if len(pcs) <= 1:
        return []
    steps = [b - a for a, b in zip(pcs, pcs[1:])]
    steps.append((pcs[0] + n) - pcs[-1])
    return steps


def max_circular_gap(pcs: Sequence[int], n: int = N_PCS_DEFAULT) -> int:
    if len(pcs) <= 1:
        return n
    return max(step_vector_from_pcs(pcs, n))


def rotate(seq: Sequence[int], tau: int) -> List[int]:
    if not seq:
        return list(seq)
    k = len(seq)
    tau %= k
    return list(seq[tau:]) + list(seq[:tau])


def reflect(seq: Sequence[int]) -> List[int]:
    return list(reversed(seq))


def canonical_step_pattern(steps: Sequence[int], dihedral: bool = False) -> Tuple[int, ...]:
    """Return canonical representative tuple under rotation (and optional reflection).

    Uses lexicographically smallest rotation/reflection.
    """
    k = len(steps)
    if k == 0:
        return tuple()
    candidates: List[Tuple[int, ...]] = []
    base = list(steps)
    for t in range(k):
        candidates.append(tuple(rotate(base, t)))
    if dihedral:
        rev = reflect(base)
        for t in range(k):
            candidates.append(tuple(rotate(rev, t)))
    return min(candidates)


def iter_scale_masks(
    n: int = N_PCS_DEFAULT,
    require_root: bool = True,
    min_k: int = 1,
    max_k: Optional[int] = None,
    max_gap: Optional[int] = None,
) -> Iterator[int]:
    """Iterate bitmasks satisfying constraints.

    - n: number of pitch classes on the circle (default 12)
    - require_root: if True, require pc 0 to be present
    - min_k/max_k: filter by set size
    - max_gap: maximum allowed circular step size between successive pcs
    """
    if max_k is None:
        max_k = n
    root_mask = 1 if require_root else 0
    full = 1 << n
    for mask in range(full):
        if require_root and (mask & 1) == 0:
            continue
        k = mask.bit_count()
        if k < min_k or k > max_k or k == 0:
            continue
        pcs = pcs_from_mask(mask, n)
        if max_gap is not None and max_circular_gap(pcs, n) > max_gap:
            continue
        yield mask


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    import math

    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    denom = na * nb
    return 0.0 if denom == 0 else float(dot / denom)


def arrangement_defect(steps: Sequence[int]) -> float:
    k = len(steps)
    if k == 0:
        return 0.0
    rev = reflect(steps)
    best = -1.0
    for t in range(k):
        sim = cosine_similarity(steps, rotate(rev, t))
        if sim > best:
            best = sim
    best = max(0.0, min(1.0, best))
    return 1.0 - best


def evenness_defect(steps: Sequence[int], n: int = N_PCS_DEFAULT) -> float:
    k = len(steps)
    if k <= 1:
        return 0.0
    mu = n / k
    diffsq_sum = 0.0
    for s in steps:
        d = float(s) - mu
        diffsq_sum += d * d
    mse = diffsq_sum / k
    import math

    std = math.sqrt(mse)
    # Normalization: upper bound occurs when one gap is large, others small.
    # For simplicity, divide by mu (dimensionless proxy). This yields [0, ~1].
    return float(std / mu)


def shannon_entropy_bits(steps: Sequence[int], n: int = N_PCS_DEFAULT) -> Tuple[float, float]:
    """Return (H, H_norm) where H is in bits and H_norm in [0,1].

    Distribution is over step sizes observed in the step vector.
    Normalization is by log2(min(k, n)).
    """
    from math import log2

    k = len(steps)
    if k == 0:
        return 0.0, 0.0
    counts = {}
    for s in steps:
        counts[s] = counts.get(s, 0) + 1
    H = 0.0
    for c in counts.values():
        p = c / k
        H -= p * log2(p)
    Hmax = log2(min(k, n)) if k > 0 else 1.0
    Hn = 0.0 if Hmax == 0 else (H / Hmax)
    if Hn < 0:
        Hn = 0.0
    if Hn > 1:
        Hn = 1.0
    return float(H), float(Hn)


def lz76_complexity_norm(seq: Sequence[int]) -> float:
    """Normalized LZ76 phrase complexity in [0,1].

    c(n) * log2(n) / n where c(n) is the number of distinct phrases.
    """
    import math

    s = list(seq)
    n = len(s)
    if n <= 1:
        return 0.0
    # LZ76 parsing
    i, k, l = 0, 1, 1
    c = 1
    while True:
        if i + k > n - 1:
            c += 1
            break
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > l:
                l = k
            i += 1
            if i == l:
                c += 1
                l += 1
                if l + 1 > n:
                    break
                i = 0
            k = 1
    cn = c * math.log2(n) / n
    return float(max(0.0, min(1.0, cn)))


def popcount(x: int) -> int:
    return x.bit_count()


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def steps_to_mask(steps: Sequence[int], n: int = N_PCS_DEFAULT) -> int:
    m = 1  # include root 0
    pc = 0
    for s in steps:
        pc = (pc + s) % n
        m |= 1 << pc
    return m


MAJOR_STEPS = (2, 2, 1, 2, 2, 2, 1)
NATURAL_MINOR_STEPS = (2, 1, 2, 2, 1, 2, 2)

MAJOR_MASK = steps_to_mask(MAJOR_STEPS, N_PCS_DEFAULT)
MINOR_MASK = steps_to_mask(NATURAL_MINOR_STEPS, N_PCS_DEFAULT)


def parse_scale_spec(spec: str, n: int = N_PCS_DEFAULT) -> int:
    """Parse scale spec like 'major', 'minor', 'mask:1234', 'steps:2-2-1-...'"""
    s = spec.strip().lower()
    if s == "major":
        return MAJOR_MASK
    if s in {"minor", "aeolian", "natural-minor"}:
        return MINOR_MASK
    if s.startswith("mask:"):
        return int(s.split(":", 1)[1])
    if s.startswith("steps:"):
        body = s.split(":", 1)[1]
        parts = [int(x) for x in body.replace(",", "-").split("-") if x.strip()]
        return steps_to_mask(parts, n)
    raise ValueError(f"Unrecognized scale spec: {spec}")


@dataclass
class Graph:
    nodes: List[int]
    edges: List[Tuple[int, int]]
    index: dict


def build_graph(
    masks: Sequence[int],
    n: int = N_PCS_DEFAULT,
    edge_type: str = "flip",
) -> Graph:
    """Build adjacency by either 'flip' (Hamming=1) or 'swap' (Hamming=2, size-preserving)."""
    nodes = list(masks)
    index = {m: i for i, m in enumerate(nodes)}
    edges: List[Tuple[int, int]] = []
    if edge_type == "flip":
        for m in nodes:
            for bit in range(n):
                nb = m ^ (1 << bit)
                if nb in index and nb > m:
                    edges.append((m, nb))
    elif edge_type == "swap":
        for m in nodes:
            k = popcount(m)
            offs = [b for b in range(n) if (m >> b) & 1]
            ons = [b for b in range(n) if not ((m >> b) & 1)]
            for b_off in offs:
                for b_on in ons:
                    nb = (m ^ (1 << b_off)) | (1 << b_on)
                    if popcount(nb) != k:
                        continue
                    if nb in index and nb > m:
                        edges.append((m, nb))
    else:
        raise ValueError("edge_type must be 'flip' or 'swap'")
    return Graph(nodes=nodes, edges=edges, index=index)


def graph_components(g: Graph) -> List[List[int]]:
    adj = {m: [] for m in g.nodes}
    for u, v in g.edges:
        adj[u].append(v)
        adj[v].append(u)
    comps: List[List[int]] = []
    seen = set()
    for m in g.nodes:
        if m in seen:
            continue
        stack = [m]
        comp = []
        seen.add(m)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        comps.append(comp)
    return comps


def shortest_path(g: Graph, src: int, dst: int) -> Optional[List[int]]:
    from collections import deque

    if src == dst:
        return [src]
    adj = {m: [] for m in g.nodes}
    for u, v in g.edges:
        adj[u].append(v)
        adj[v].append(u)
    q = deque([src])
    prev = {src: None}
    while q:
        x = q.popleft()
        for y in adj[x]:
            if y in prev:
                continue
            prev[y] = x
            if y == dst:
                # reconstruct
                path = [dst]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                path.reverse()
                return path
            q.append(y)
    return None

