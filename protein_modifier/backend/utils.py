"""
Docstring for protein_modifier.backend.utils
"""
from __future__ import annotations

import tempfile
import os
import mdtraj as md
from protein_modifier.backend.data_structures import Structure, Atom, Residue, Chain
from protein_modifier.backend.io import parse_cif, write_cif, write_pdb, parse_pdb

def get_sasa_by_residue(structure: Structure, structure_file: str, probe_radius: float = 1.4) -> Structure:
    if structure_file.lower().endswith('.cif'):
        # MDTraj uses OpenMM to load CIF files, which adds a heavy, unnecessary dependency.
        # As a workaround, we write a temporary PDB file to calculate SASA.
        fd, tmp_path = tempfile.mkstemp(suffix='.pdb')
        os.close(fd)
        try:
            write_pdb(structure.to_dict(), tmp_path)
            traj = md.load(tmp_path)
        finally:
            os.remove(tmp_path)
    else:
        traj = md.load(structure_file)
        
    sasa_values = 100*md.shrake_rupley(traj, mode='residue', probe_radius=probe_radius*0.1)
    # flatten
    sasa_values = sasa_values.flatten()
    # is a numpy array. Use numpy conventions to assign 1 if SASA is less than 20 (buried) and 0 if SASA is greater than or equal to 20 (exposed).
    sasa_values = (sasa_values < 20).astype(int)
    sasa_values = sasa_values.tolist()
    structure.assign_residue_solvent_access(sasa_values)
    return structure