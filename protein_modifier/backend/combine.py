"""Utilities for combining two structures into one output file."""
from __future__ import annotations

import logging
import os

from protein_modifier.backend.data_structures import Structure
from protein_modifier.backend.io import parse_structure, write_cif, write_pdb

logger = logging.getLogger(__name__)


def combine_structures(
    first_path: str,
    second_path: str,
    output_path: str,
    rename_chains: bool = True,
) -> dict[str, int | list[str]]:
    """Combine two structures and write them to one PDB or mmCIF.

    The second structure is merged into the first. Chain-ID collisions are
    automatically resolved by renaming chains from the second structure when
    ``rename_chains`` is True. Atom serial IDs are renumbered before writing.

    Parameters
    ----------
    first_path : str
        Path to the first input structure (.pdb or .cif). Its non-colliding
        chain IDs are preserved.
    second_path : str
        Path to the second input structure (.pdb or .cif). Any colliding chain
        IDs are renamed if ``rename_chains`` is True.
    output_path : str
        Path to the combined output (.pdb/.ent or .cif/.mmcif).
    rename_chains : bool
        If True, automatically rename colliding chain IDs from ``second_path``.

    Returns
    -------
    dict
        Summary with ``n_chains``, ``n_atoms``, and ``chains``.
    """
    logger.info(f"Loading first structure:  {first_path}")
    first = Structure.from_dict(parse_structure(first_path), name="first")
    logger.info(f"Loading second structure: {second_path}")
    second = Structure.from_dict(parse_structure(second_path), name="second")

    combined = first.merge(second, rename_chains=rename_chains)
    combined.need_to_update_atom_numbers = True

    ext = os.path.splitext(output_path)[1].lower()
    if ext in ('.cif', '.mmcif'):
        write_cif(combined.to_dict(), output_path)
    elif ext in ('.pdb', '.ent'):
        write_pdb(combined.to_dict(), output_path)
    else:
        raise ValueError(
            f"Unsupported output extension '{ext}'. Use .cif/.mmcif or .pdb/.ent."
        )

    n_atoms = sum(
        1
        for chain in combined.chains.values()
        for residue in chain
        for _atom in residue
    )
    chains = sorted(combined.chains)
    logger.info(
        f"Wrote combined structure with {len(chains)} chains and {n_atoms} atoms to {output_path}"
    )
    return {
        'n_chains': len(chains),
        'n_atoms': n_atoms,
        'chains': chains,
    }