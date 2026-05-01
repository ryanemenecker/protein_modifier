"""
Robust CA-based structural alignment of two protein structures.

Aligns a "mobile" structure onto a "reference" structure using only their
alpha carbon coordinates. Correspondence between CAs is established
**purely structurally** — by iterative-closest-point (ICP) on CA point
clouds with a PCA-based initial pose search. Amino-acid identities and
residue numbering are *not* used at any stage.

The resulting rigid-body transform (rotation + translation, found by
Kabsch / SVD over the converged correspondences) is then applied to
**every atom** of the mobile structure, not just CAs.
"""
from __future__ import annotations

import logging
import itertools

import numpy as np

from protein_modifier.backend.data_structures import Structure, Chain, Residue, Atom
from protein_modifier.backend.io import parse_structure, write_cif, write_pdb
from protein_modifier.backend.modify_structure import kabsch_align
from protein_modifier.data.amino_acids import (
    AA_MAP_3_TO_1,
    NONSTANDARD_AA_MAP_3_TO_1,
)

logger = logging.getLogger(__name__)

VALID_METHODS = ("structure", "sequence")


def _ca_residues(chain: Chain) -> list[tuple[Residue, Atom]]:
    """Sorted list of (residue, CA atom) for residues that contain a CA."""
    out: list[tuple[Residue, Atom]] = []
    for res in chain.get_sorted_residues():
        try:
            ca = res['CA']
        except KeyError:
            continue
        out.append((res, ca))
    return out


def _residue_one_letter(res_name: str) -> str:
    """Return the 1-letter code for ``res_name`` or 'X' if unknown."""
    key = (res_name or '').strip().upper()
    if key in AA_MAP_3_TO_1:
        return AA_MAP_3_TO_1[key]
    if key in NONSTANDARD_AA_MAP_3_TO_1:
        return NONSTANDARD_AA_MAP_3_TO_1[key]
    return 'X'


def _sequence_pairs(
    ref_seq: str,
    mob_seq: str,
    match: float = 2.0,
    mismatch: float = -1.0,
    gap: float = -2.0,
) -> tuple[np.ndarray, float]:
    """Needleman-Wunsch global alignment of two 1-letter sequences.

    Returns
    -------
    pairs : (k, 2) ndarray of int
        ``(i_mob, j_ref)`` index pairs for matched columns (no gap-aligned
        columns are returned). Order is N->C along both sequences.
    score : float
        The optimal global-alignment score (higher = better).
    """
    n = len(mob_seq)
    m = len(ref_seq)
    if n == 0 or m == 0:
        return np.empty((0, 2), dtype=np.int64), 0.0

    score = np.zeros((n + 1, m + 1), dtype=np.float64)
    trace = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0 diag, 1 up (gap-ref), 2 left (gap-mob)
    score[1:, 0] = np.arange(1, n + 1) * gap
    score[0, 1:] = np.arange(1, m + 1) * gap
    trace[1:, 0] = 1
    trace[0, 1:] = 2

    for i in range(1, n + 1):
        a = mob_seq[i - 1]
        for j in range(1, m + 1):
            b = ref_seq[j - 1]
            s = match if (a == b and a != 'X') else mismatch
            diag = score[i - 1, j - 1] + s
            up = score[i - 1, j] + gap
            left = score[i, j - 1] + gap
            if diag >= up and diag >= left:
                score[i, j] = diag
                trace[i, j] = 0
            elif up >= left:
                score[i, j] = up
                trace[i, j] = 1
            else:
                score[i, j] = left
                trace[i, j] = 2

    pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        t = trace[i, j]
        if t == 0:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif t == 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return (
        np.array(pairs, dtype=np.int64) if pairs else np.empty((0, 2), dtype=np.int64),
        float(score[n, m]),
    )


def _select_chains(structure: Structure, chain_id: str | None) -> list[Chain]:
    if chain_id is not None:
        if chain_id not in structure.chains:
            raise ValueError(
                f"Chain '{chain_id}' not found in {structure.name}. "
                f"Available: {list(structure.chains)}"
            )
        return [structure.chains[chain_id]]
    return list(structure.chains.values())


# ──────────────────────────────────────────────
# Structural CA-cloud alignment (ICP + PCA init)
# ──────────────────────────────────────────────

def _pca_axes(coords_centered: np.ndarray) -> np.ndarray:
    """Return a (3, 3) matrix whose rows are the principal axes (descending
    variance) of the centered coordinates."""
    # SVD of (N, 3) gives V^T (3, 3); rows of Vt are principal axes.
    _, _, Vt = np.linalg.svd(coords_centered, full_matrices=False)
    return Vt


def _icp_fit(
    mobile: np.ndarray,
    target: np.ndarray,
    max_iter: int = 60,
    tol: float = 1e-5,
    gap_penalty: float = 6.0,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Iterative closest point on two CA clouds with order-preserving
    correspondence.

    Both arrays are assumed to be already in a common rough frame (the
    caller provides an initial pose by transforming ``mobile``). The CA
    coordinates are taken to be in N→C chain order — a purely structural
    property of the polypeptide backbone, independent of residue
    identity. The correspondence step is therefore order-preserving:
    Needleman–Wunsch-style DP over Euclidean distances, with a gap
    penalty (Å) that lets unmatched residues drop out rather than being
    forced onto a wrong partner. This avoids the failure mode of plain
    nearest-neighbour ICP, where multiple mobile points snap onto the
    same target.

    Returns
    -------
    rmsd : float
        RMSD over the converged matched pairs (Å).
    R : (3, 3) ndarray
    t : (3,) ndarray
        ``R``, ``t`` map *original mobile points* to the target frame:
        ``aligned = mobile @ R.T + t``.
    correspondence : ndarray of shape (k, 2), int
        ``correspondence[k] = (i_mobile, j_target)`` for each matched pair,
        in chain order.
    """
    R_total = np.eye(3)
    t_total = np.zeros(3)
    current = mobile.copy()
    prev_rmsd = float('inf')
    pairs = np.empty((0, 2), dtype=np.int64)

    for _ in range(max_iter):
        pairs = _order_preserving_pairs(current, target, gap_penalty)
        if pairs.shape[0] < 3:
            break
        mob_sel = current[pairs[:, 0]]
        tar_sel = target[pairs[:, 1]]
        rmsd = float(np.sqrt(((mob_sel - tar_sel) ** 2).sum(axis=1).mean()))

        R_step, mob_c, tar_c = kabsch_align(mob_sel, tar_sel)
        current = (current - mob_c) @ R_step.T + tar_c

        # Compose into the running transform.
        t_total = R_step @ (t_total - mob_c) + tar_c
        R_total = R_step @ R_total

        if abs(prev_rmsd - rmsd) < tol:
            break
        prev_rmsd = rmsd

    pairs = _order_preserving_pairs(current, target, gap_penalty)
    if pairs.shape[0] >= 1:
        mob_sel = current[pairs[:, 0]]
        tar_sel = target[pairs[:, 1]]
        rmsd = float(np.sqrt(((mob_sel - tar_sel) ** 2).sum(axis=1).mean()))
    else:
        rmsd = float('inf')
    return rmsd, R_total, t_total, pairs


def _order_preserving_pairs(
    mobile: np.ndarray, target: np.ndarray, gap_penalty: float
) -> np.ndarray:
    """Optimal order-preserving correspondence between two ordered point
    sequences via Needleman–Wunsch on Euclidean distances.

    Returns an ``(k, 2)`` array of ``(i_mobile, j_target)`` index pairs
    that minimize the total cost
        ``sum(||mobile[i] - target[j]||) + gap_penalty * (n_gaps)``
    subject to ``i`` and ``j`` both being strictly increasing along the
    returned list. Uses only coordinates — no residue labels.
    """
    n = mobile.shape[0]
    m = target.shape[0]
    if n == 0 or m == 0:
        return np.empty((0, 2), dtype=np.int64)

    # Pairwise Euclidean distances, shape (n, m).
    diff = mobile[:, None, :] - target[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=-1))

    # DP: score[i, j] = min cost aligning mobile[:i] with target[:j].
    # Lower is better; gap aligns one side to nothing at cost gap_penalty.
    INF = np.inf
    score = np.full((n + 1, m + 1), INF, dtype=np.float64)
    trace = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0=match, 1=gap-mob, 2=gap-tar
    score[0, 0] = 0.0
    score[1:, 0] = np.arange(1, n + 1) * gap_penalty
    score[0, 1:] = np.arange(1, m + 1) * gap_penalty
    trace[1:, 0] = 1
    trace[0, 1:] = 2

    for i in range(1, n + 1):
        di = dist[i - 1]
        prev_row = score[i - 1]
        cur_row = score[i]
        cur_trace = trace[i]
        for j in range(1, m + 1):
            match = prev_row[j - 1] + di[j - 1]
            gap_m = prev_row[j] + gap_penalty
            gap_t = cur_row[j - 1] + gap_penalty
            if match <= gap_m and match <= gap_t:
                cur_row[j] = match
                cur_trace[j] = 0
            elif gap_m <= gap_t:
                cur_row[j] = gap_m
                cur_trace[j] = 1
            else:
                cur_row[j] = gap_t
                cur_trace[j] = 2

    # Traceback.
    pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        t = trace[i, j]
        if t == 0:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif t == 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return np.array(pairs, dtype=np.int64) if pairs else np.empty((0, 2), dtype=np.int64)


def _structural_align_clouds(
    ref_coords: np.ndarray, mob_coords: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Align two CA point clouds purely structurally.

    Strategy
    --------
    1. Centre both clouds on their centroids.
    2. Compute principal axes (PCA) for each cloud.
    3. Build a pool of candidate initial rotations:
         * the 4 proper-rotation sign combinations of the mobile principal
           axes onto the reference principal axes,
         * a number of pseudo-random rotations (deterministic seed) so
           that PCA-degenerate or near-symmetric clouds still get a good
           cover of orientation space.
    4. Run ICP-with-DP-correspondence from each candidate. Keep the best
       (lowest RMSD) result.

    Correspondences are order-preserving along the chain — a structural
    constraint of the polypeptide backbone — and are computed without any
    reference to residue identity or numbering.

    Returns ``(rmsd, R, t, correspondence)`` mapping the *original* mobile
    coordinates to the reference frame: ``aligned = mob_coords @ R.T + t``.
    Each row of ``correspondence`` is ``(i_mobile, j_ref)``.
    """
    ref_centroid = ref_coords.mean(axis=0)
    mob_centroid = mob_coords.mean(axis=0)
    ref_c = ref_coords - ref_centroid
    mob_c = mob_coords - mob_centroid

    Vt_ref = _pca_axes(ref_c)
    Vt_mob = _pca_axes(mob_c)

    initial_rotations: list[np.ndarray] = []

    # 4 PCA sign combinations giving proper rotations.
    for signs in itertools.product([1, -1], repeat=3):
        S = np.diag(signs).astype(float)
        R_init = Vt_ref.T @ S @ Vt_mob
        if np.linalg.det(R_init) > 0:
            initial_rotations.append(R_init)

    # Add deterministic pseudo-random rotations to escape PCA-only failures.
    rng = np.random.default_rng(0xC0FFEE)
    for _ in range(24):
        # Uniformly sampled rotation via QR of a Gaussian matrix.
        A = rng.standard_normal((3, 3))
        Q, Rm = np.linalg.qr(A)
        Q = Q @ np.diag(np.sign(np.diag(Rm)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        initial_rotations.append(Q)

    best: tuple[float, np.ndarray, np.ndarray, np.ndarray] | None = None
    for R_init in initial_rotations:
        # Apply initial pose to mobile-centered cloud, then bring into ref frame.
        mob_init = mob_c @ R_init.T + ref_centroid
        rmsd, R_icp, t_icp, pairs = _icp_fit(mob_init, ref_coords)

        # Compose initial pose with ICP's transform so the result maps the
        # *original* mobile coordinates: x'  = (x - mob_centroid) R_init^T + ref_centroid,
        # then x'' = x' R_icp^T + t_icp.
        R_total = R_icp @ R_init
        t_total = R_icp @ (ref_centroid - R_init @ mob_centroid) + t_icp

        if best is None or rmsd < best[0]:
            best = (rmsd, R_total, t_total, pairs)

    if best is None:  # unreachable in practice
        rmsd, R_icp, t_icp, pairs = _icp_fit(mob_coords.copy(), ref_coords)
        best = (rmsd, R_icp, t_icp, pairs)
    return best


def _matched_ca_pairs(
    ref_chain: Chain,
    mob_chain: Chain,
    method: str = "structure",
) -> tuple[list[tuple[Residue, Atom, Residue, Atom]], np.ndarray, np.ndarray]:
    """Establish CA correspondences between two chains.

    Parameters
    ----------
    method : {'structure', 'sequence'}
        ``'structure'`` (default) — pure-structural ICP+DP on CA clouds;
        ignores residue identity. ``'sequence'`` — Needleman-Wunsch on
        1-letter residue sequences, then Kabsch on the matched CAs;
        appropriate when both inputs are the same (or homologous) protein.

    Returns
    -------
    matches : list of (ref_residue, ref_CA, mob_residue, mob_CA)
        One pair per matched residue.
    R, t : ndarrays
        Rigid transform mapping original mobile CA coordinates into the
        reference frame: ``aligned = mob_coords @ R.T + t``.
    """
    if method not in VALID_METHODS:
        raise ValueError(f"method must be one of {VALID_METHODS}; got {method!r}")

    ref_pairs = _ca_residues(ref_chain)
    mob_pairs = _ca_residues(mob_chain)
    if len(ref_pairs) < 3 or len(mob_pairs) < 3:
        return [], np.eye(3), np.zeros(3)

    ref_coords = np.array([[ca.x, ca.y, ca.z] for _, ca in ref_pairs], dtype=np.float64)
    mob_coords = np.array([[ca.x, ca.y, ca.z] for _, ca in mob_pairs], dtype=np.float64)

    if method == "sequence":
        ref_seq = ''.join(_residue_one_letter(r.name) for r, _ in ref_pairs)
        mob_seq = ''.join(_residue_one_letter(r.name) for r, _ in mob_pairs)
        pairs, _ = _sequence_pairs(ref_seq, mob_seq)
        if pairs.shape[0] < 3:
            return [], np.eye(3), np.zeros(3)
        mob_sel = mob_coords[pairs[:, 0]]
        ref_sel = ref_coords[pairs[:, 1]]
        R_k, mob_c, ref_c = kabsch_align(mob_sel, ref_sel)
        # transform original mobile coords -> reference frame
        R = R_k
        t = ref_c - R_k @ mob_c
    else:
        _, R, t, pairs = _structural_align_clouds(ref_coords, mob_coords)

    matched: list[tuple[Residue, Atom, Residue, Atom]] = []
    for i_mob, j_ref in pairs:
        m_res, m_ca = mob_pairs[int(i_mob)]
        r_res, r_ca = ref_pairs[int(j_ref)]
        matched.append((r_res, r_ca, m_res, m_ca))
    return matched, R, t


def _pair_chains(
    ref_chains: list[Chain],
    mob_chains: list[Chain],
    method: str = "structure",
) -> list[tuple[Chain, Chain]]:
    """Pair reference and mobile chains for alignment.

    With ``method='structure'`` chain pairing is purely structural (ICP+DP
    RMSD on CA clouds). With ``method='sequence'`` chain pairing uses the
    Needleman-Wunsch score on the 1-letter residue sequences (higher score =
    better pairing).
    """
    if method not in VALID_METHODS:
        raise ValueError(f"method must be one of {VALID_METHODS}; got {method!r}")

    ref_chains = [c for c in ref_chains if _ca_residues(c)]
    mob_chains = [c for c in mob_chains if _ca_residues(c)]
    if not ref_chains or not mob_chains:
        return []

    def structural_score(r: Chain, m: Chain) -> float:
        r_pairs = _ca_residues(r)
        m_pairs = _ca_residues(m)
        if len(r_pairs) < 3 or len(m_pairs) < 3:
            return float('inf')
        rc = np.array([[ca.x, ca.y, ca.z] for _, ca in r_pairs], dtype=np.float64)
        mc = np.array([[ca.x, ca.y, ca.z] for _, ca in m_pairs], dtype=np.float64)
        rmsd, _, _, _ = _structural_align_clouds(rc, mc)
        # Penalize size mismatch slightly so a small mobile chain does not
        # spuriously fit inside a larger reference chain at near-zero RMSD
        # (only relevant for multi-vs-multi pairing tie-breaks).
        size_ratio = min(len(r_pairs), len(m_pairs)) / max(len(r_pairs), len(m_pairs))
        return rmsd / max(size_ratio, 1e-3)

    def sequence_score(r: Chain, m: Chain) -> float:
        # Lower-is-better convention to match structural_score: return -NW.
        r_pairs = _ca_residues(r)
        m_pairs = _ca_residues(m)
        if len(r_pairs) < 3 or len(m_pairs) < 3:
            return float('inf')
        r_seq = ''.join(_residue_one_letter(rr.name) for rr, _ in r_pairs)
        m_seq = ''.join(_residue_one_letter(rr.name) for rr, _ in m_pairs)
        _, score = _sequence_pairs(r_seq, m_seq)
        return -score

    fit_score = sequence_score if method == "sequence" else structural_score

    # Single-chain on one side: broadcast to its best structural match.
    if len(ref_chains) == 1 and len(mob_chains) >= 1:
        scored = [(fit_score(ref_chains[0], m), m) for m in mob_chains]
        scored.sort(key=lambda x: x[0])
        return [(ref_chains[0], scored[0][1])]
    if len(mob_chains) == 1 and len(ref_chains) >= 1:
        scored = [(fit_score(r, mob_chains[0]), r) for r in ref_chains]
        scored.sort(key=lambda x: x[0])
        return [(scored[0][1], mob_chains[0])]

    # Multi-vs-multi: greedy structural matching.
    pairs: list[tuple[Chain, Chain]] = []
    ref_remaining = list(ref_chains)
    mob_remaining = list(mob_chains)
    while ref_remaining and mob_remaining:
        best = (float('inf'), None, None)
        for r in ref_remaining:
            for m in mob_remaining:
                s = fit_score(r, m)
                if s < best[0]:
                    best = (s, r, m)
        _, r_best, m_best = best
        if r_best is None or m_best is None:
            break
        pairs.append((r_best, m_best))
        ref_remaining.remove(r_best)
        mob_remaining.remove(m_best)
    return pairs


def _flatten_to_chain(chains: list[Chain], new_chain_id: str) -> Chain:
    """Combine all residues from ``chains`` into a single new ``Chain``.

    Residue IDs are preserved when unique across all input chains, otherwise
    renumbered sequentially. All atom coordinates and metadata are kept;
    only the chain ID (and, where renumbered, the residue ID) are updated.
    """
    seen: set[str] = set()
    collide = False
    for ch in chains:
        for rid in ch.residues:
            if rid in seen:
                collide = True
                break
            seen.add(rid)
        if collide:
            break

    new_chain = Chain(new_chain_id)
    counter = 1
    for ch in chains:
        for residue in ch.get_sorted_residues():
            new_rid = str(counter) if collide else residue.id
            new_res = Residue(new_rid, [], residue.name, new_chain_id)
            new_res.was_built = residue.was_built
            for atom in residue.atoms:
                d = dict(atom.data)
                d['label_asym_id'] = new_chain_id
                d['auth_asym_id'] = new_chain_id
                d['label_seq_id'] = new_rid
                d['auth_seq_id'] = new_rid
                # Sync live coordinates from the source Atom onto the data
                # dict before constructing a new Atom — Atom.__init__ reads
                # x/y/z from data, so without this any rigid-body transform
                # applied via atom.x/y/z (e.g. the alignment fit) would be
                # silently discarded in the output.
                d['Cartn_x'] = f"{atom.x:.3f}"
                d['Cartn_y'] = f"{atom.y:.3f}"
                d['Cartn_z'] = f"{atom.z:.3f}"
                new_res.atoms.append(Atom(d))
            new_chain.residues[new_rid] = new_res
            counter += 1
    return new_chain


def _build_ca_only_chain(
    ca_pairs: list[tuple[Residue, Atom]], new_chain_id: str
) -> Chain:
    """Build a chain containing only the given CA atoms, renumbered 1..N.

    Residue names are taken from the source residues; coordinates are read
    live from the ``Atom`` objects, so any rigid-body transform applied to
    those atoms before this is called is reflected in the output.
    """
    new_chain = Chain(new_chain_id)
    for i, (src_res, src_ca) in enumerate(ca_pairs, start=1):
        new_rid = str(i)
        new_res = Residue(new_rid, [], src_res.name, new_chain_id)
        d = dict(src_ca.data)
        d['label_asym_id'] = new_chain_id
        d['auth_asym_id'] = new_chain_id
        d['label_seq_id'] = new_rid
        d['auth_seq_id'] = new_rid
        d['label_atom_id'] = 'CA'
        d['auth_atom_id'] = 'CA'
        d['type_symbol'] = 'C'
        d['Cartn_x'] = f"{src_ca.x:.3f}"
        d['Cartn_y'] = f"{src_ca.y:.3f}"
        d['Cartn_z'] = f"{src_ca.z:.3f}"
        new_res.atoms.append(Atom(d))
        new_chain.residues[new_rid] = new_res
    return new_chain


def align_structures(
    reference_path: str,
    mobile_path: str,
    output_path: str,
    ref_chain: str | None = None,
    mobile_chain: str | None = None,
    ca_only: bool = False,
    method: str = "structure",
) -> dict:
    """
    Align ``mobile_path`` onto ``reference_path`` by alpha carbons and write a
    combined structure with the reference as chain ``A`` and the aligned
    mobile structure as chain ``B``.

    The alignment is **purely structural**: CA-to-CA correspondence is
    found by iterative-closest-point (ICP) on CA point clouds with a
    PCA-based initial-pose search. Amino-acid identities and residue
    numbering are *not* consulted. Once the correspondence is determined,
    the optimal rigid-body transform that minimizes RMSD is computed via
    the Kabsch algorithm and applied to **all** atoms of the mobile
    structure.

    Parameters
    ----------
    reference_path : str
        Path to the reference structure (.cif or .pdb).
    mobile_path : str
        Path to the structure to be aligned (.cif or .pdb).
    output_path : str
        Path to write combined output. Format inferred from extension
        (.cif / .mmcif → mmCIF; .pdb / .ent → PDB).
    ref_chain : str, optional
        Restrict the reference to a single chain ID. Default: use all chains.
    mobile_chain : str, optional
        Restrict the mobile structure to a single chain ID. Default: all chains.
    ca_only : bool
        If True, write only the matched CA atoms used for the alignment
        (reference CAs as chain A, mobile CAs as chain B), all in the
        post-fit frame. If False (default), write every atom of the
        selected chains (sidechains, cofactors, ions, etc.).
    method : {'structure', 'sequence'}
        How to find CA correspondences. ``'structure'`` (default) is pure
        ICP on CA clouds (ignores residue identity); use this when the
        two structures are of unrelated or unknown sequences, or when
        residue numbering is unreliable. ``'sequence'`` runs Needleman-
        Wunsch on 1-letter residue sequences and then Kabsch on the
        matched CAs; use this when the inputs are the same protein (or
        close homologs) — it is faster and more robust to large
        conformational change.

    Returns
    -------
    dict
        Keys: ``rmsd`` (Å, over matched CA atoms), ``n_matched`` (int),
        ``rotation`` (3x3 ndarray), ``translation`` (3-vector ndarray).

    Raises
    ------
    ValueError
        If fewer than 3 matched CA pairs can be found, or the output
        extension is not recognized.
    """
    if method not in VALID_METHODS:
        raise ValueError(f"method must be one of {VALID_METHODS}; got {method!r}")

    logger.info(f"Loading reference structure: {reference_path}")
    ref_struct = Structure.from_dict(parse_structure(reference_path), name="reference")
    logger.info(f"Loading mobile structure:    {mobile_path}")
    mob_struct = Structure.from_dict(parse_structure(mobile_path), name="mobile")

    ref_chains = _select_chains(ref_struct, ref_chain)
    mob_chains = _select_chains(mob_struct, mobile_chain)

    logger.info(f"Alignment method: {method}")
    chain_pairs = _pair_chains(ref_chains, mob_chains, method=method)
    logger.info(
        "Chain pairing: "
        + ", ".join(f"{r.id}<->{m.id}" for r, m in chain_pairs)
    )

    # Per-chain matched CA records: keep references to the live Atom objects
    # so post-fit coordinates flow through automatically when we write output.
    # Correspondences are derived purely structurally (PCA + ICP on CA clouds).
    matched_per_pair: list[tuple[Chain, Chain, list[tuple[Residue, Atom, Residue, Atom]]]] = []
    ref_coords_all: list[list[float]] = []
    mob_coords_all: list[list[float]] = []
    for ref_c, mob_c in chain_pairs:
        matches, _R_chain, _t_chain = _matched_ca_pairs(ref_c, mob_c, method=method)
        logger.info(
            f"  chain {ref_c.id} <-> {mob_c.id}: matched {len(matches)} CA pairs"
        )
        matched_per_pair.append((ref_c, mob_c, matches))
        for _, r_ca, _, m_ca in matches:
            ref_coords_all.append([r_ca.x, r_ca.y, r_ca.z])
            mob_coords_all.append([m_ca.x, m_ca.y, m_ca.z])

    n_matched = len(ref_coords_all)
    if n_matched < 3:
        raise ValueError(
            f"Need at least 3 matched CA atoms for alignment; found {n_matched}. "
            "Check that the inputs contain CA atoms in both structures."
        )

    ref_arr = np.asarray(ref_coords_all, dtype=np.float64)
    mob_arr = np.asarray(mob_coords_all, dtype=np.float64)

    # Optimal rigid-body fit (Kabsch / SVD) — minimizes RMSD.
    R, mob_centroid, ref_centroid = kabsch_align(mob_arr, ref_arr)

    # Apply transform to every atom of the mobile structure (not just CAs)
    # so sidechains / ligands move rigidly with the protein.
    for chain in mob_struct.chains.values():
        for residue in chain:
            for atom in residue:
                p = np.array([atom.x, atom.y, atom.z], dtype=np.float64) - mob_centroid
                p_rot = R @ p + ref_centroid
                atom.x = float(p_rot[0])
                atom.y = float(p_rot[1])
                atom.z = float(p_rot[2])

    aligned_mob = (mob_arr - mob_centroid) @ R.T + ref_centroid
    rmsd = float(np.sqrt(((aligned_mob - ref_arr) ** 2).sum(axis=1).mean()))
    logger.info(f"Alignment RMSD over {n_matched} CA atoms: {rmsd:.3f} A")

    combined = Structure(name="aligned")
    if ca_only:
        # CA-only output: just the matched CAs, post-fit, ref->A, mob->B.
        # Residues are renumbered 1..N along the matched sequence so chain A
        # and chain B are 1:1 by residue index.
        ref_ca_pairs: list[tuple[Residue, Atom]] = []
        mob_ca_pairs: list[tuple[Residue, Atom]] = []
        for _, _, matches in matched_per_pair:
            for r_res, r_ca, m_res, m_ca in matches:
                ref_ca_pairs.append((r_res, r_ca))
                mob_ca_pairs.append((m_res, m_ca))
        combined.chains["A"] = _build_ca_only_chain(ref_ca_pairs, "A")
        combined.chains["B"] = _build_ca_only_chain(mob_ca_pairs, "B")
    else:
        # Full output: every selected chain on each side, post-fit.
        combined.chains["A"] = _flatten_to_chain(ref_chains, "A")
        combined.chains["B"] = _flatten_to_chain(mob_chains, "B")
    combined.need_to_update_atom_numbers = True

    ext = output_path.rsplit('.', 1)[-1].lower() if '.' in output_path else ''
    if ext in ('cif', 'mmcif'):
        write_cif(combined.to_dict(), output_path)
    elif ext in ('pdb', 'ent'):
        write_pdb(combined.to_dict(), output_path)
    else:
        raise ValueError(
            f"Unsupported output extension '.{ext}'. Use .cif/.mmcif or .pdb/.ent."
        )

    logger.info(f"Wrote aligned structure to {output_path}")
    return {
        'rmsd': rmsd,
        'n_matched': n_matched,
        'rotation': R,
        'translation': ref_centroid - R @ mob_centroid,
    }
