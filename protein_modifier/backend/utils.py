"""
Docstring for protein_modifier.backend.utils
"""

import mdtraj as md
from protein_modifier.backend.data_structures import Structure, Atom, Residue, Chain
from protein_modifier.backend.io import parse_cif, write_cif, write_pdb, parse_pdb

def get_sasa_by_residue(structure, structure_file, probe_radius=1.4):
    traj = md.load(structure_file)
    sasa_values = 100*md.shrake_rupley(traj, mode='residue', probe_radius=probe_radius*0.1)
    # flatten
    sasa_values = sasa_values.flatten()
    # is a numpy array. Use numpy conventions to assign 1 if SASA is less than 20 (buried) and 0 if SASA is greater than or equal to 20 (exposed).
    sasa_values = (sasa_values < 20).astype(int)
    sasa_values = sasa_values.tolist()
    structure.assign_residue_solvent_access(sasa_values)
    return structure