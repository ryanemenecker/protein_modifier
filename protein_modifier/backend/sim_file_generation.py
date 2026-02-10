import os
import numpy as np
from protein_modifier.backend.protein_math import calculate_distance
from protein_modifier.backend.utils import get_sasa_by_residue
from protein_modifier.backend.data_structures import Atom, Residue, Chain, Structure
from protein_modifier.backend.io import parse_cif, write_cif
from protein_modifier.data.lammps_params import AA3_TO_IDX, MASSES, CHARGES

def assign_bead_type(structure, structure_file, probe_radius=1.4):
    structure = get_sasa_by_residue(structure, structure_file, probe_radius)
    for chain in structure.chains:
        for residue in structure.chains[chain].residues:
            aa_name = structure.chains[chain].residues[residue].name
            try:
                bead_type = AA3_TO_IDX[aa_name]
            except: 
                raise ValueError(f"Unknown amino acid type {aa_name}")
            # now based on aa type and solvent accessibility, assign bead type.
            if structure.chains[chain].residues[residue].solvent_accessibility is None:
                raise ValueError(f"Residue {residue.id} in chain {chain.id} does not have solvent accessibility assigned.")
            else:
                if structure.chains[chain].residues[residue].solvent_accessibility == 1:
                    # bead ids for buried residues are +20. 
                    bead_id = bead_type + 20
                else:
                    bead_id = bead_type
            structure.chains[chain].residues[residue].assign_bead_type(bead_id)
    return structure    

def generate_connect_lines(structure, bond_type=1, warn_by_dist=True, dist_thresh=6):
    # bond type is always 1. 
    bonds = []
    # track bond number for lammps file
    bond_num = 1
    # iterate over chains
    for chain in structure.chains:
        # get all residues. 
        all_residues = structure.chains[chain].residues
        all_residues = (list(all_residues.keys()))
        all_residues = [int(i) for i in all_residues]
        all_residues.sort()
        # for each residue in the chain, make it connected to the next residue. 
        for i in range(len(structure.chains[chain].residues)-1):

            res1 = structure.chains[chain].residues[str(all_residues[i])]
            res2 = structure.chains[chain].residues[str(all_residues[i+1])]
            # get atom ids for the two residues (should only be one atom each since this is coarse-grained)
            atom1_id = res1.atoms[0].data['id']
            atom2_id = res2.atoms[0].data['id']
            coords_1 = np.array((res1.atoms[0].x, res1.atoms[0].y, res1.atoms[0].z))
            coords_2 = np.array((res2.atoms[0].x, res2.atoms[0].y, res2.atoms[0].z))
            dist = calculate_distance(coords_1, coords_2)
            if warn_by_dist and dist > dist_thresh:
                print(f"Warning: Distance between residue {res1} and {res2} in chain {chain} is {dist:.2f} Angstroms, which exceeds the threshold of {dist_thresh} Angstroms. This may indicate a problem with the structure or the assigned bead types.")
            # calculate distance between the
            bonds.append(f"{bond_num} {bond_type} {atom1_id} {atom2_id}")
            bond_num += 1
    return bonds

def write_seq_dat(structure_file_path, output_path, boxdims=800):
    structure = Structure.from_dict(parse_cif(structure_file_path))
    # center strucutre
    structure.center_structure_in_box(box_size=boxdims)
    # assign bead type
    structure = assign_bead_type(structure, structure_file_path)
    # get bond info
    bond_lines = generate_connect_lines(structure)
    # get number atoms
    num_atoms = sum([len(structure.chains[chain].residues) for chain in structure.chains])
    # get number bonds
    num_bonds = len(bond_lines)
    # make base_file string
    output_str  ="LAMMPS data file for IDPs\n\n"
    output_str += f"{num_atoms} atoms\n"
    output_str += f"{num_bonds} bonds\n\n"
    output_str += "75 atom types\n"
    output_str += "1 bond types\n\n"
    output_str += f"0.0 {boxdims}   xlo xhi\n"
    output_str += f"0.0 {boxdims}   ylo yhi\n"
    output_str += f"0.0 {boxdims}   zlo zhi\n\n"
    output_str += "Masses\n\n"
    for i in range(1, 76):
        mass = MASSES[i-1]
        output_str += f"   {i} {mass:.6f}\n"
    output_str += "\nAtoms\n\n"
    atom_id = 1
    for chain in structure.chains:
        for residue in structure.chains[chain].residues:
            bead_type = structure.chains[chain].residues[residue].bead_type
            charge = CHARGES[structure.chains[chain].residues[residue].name]
            # get x coord
            x = round(structure.chains[chain].residues[residue].atoms[0].x, 3)
            y = round(structure.chains[chain].residues[residue].atoms[0].y, 3)
            z = round(structure.chains[chain].residues[residue].atoms[0].z, 3)
            output_str += f"{atom_id} 0 {bead_type} {charge} {x} {y} {z}\n"
            atom_id += 1
    output_str += "\nBonds\n\n"
    for line in bond_lines:
        output_str += line + "\n"
    with open(output_path, 'w') as f:
        f.write(output_str)
    f.close()

def find_string_indices_for_infile(structure, target_string):
    aa_string = structure.get_full_sequence()
    indices = []
    start_index = 0
    while True:
        # searching from start_index
        idx = aa_string.find(target_string, start_index)
        
        if idx == -1:
            break
            
        indices.append([idx, idx+len(target_string)-1])
        # Move past the last found index for the next search
        start_index = idx + 1 

    return indices

