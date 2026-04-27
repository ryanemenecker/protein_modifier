from __future__ import annotations

import os
import random
import logging
import numpy as np

logger = logging.getLogger(__name__)
from protein_modifier.backend.find_missing_res import get_missing_residues_by_number
from protein_modifier.backend.parse_build_file import read_build_file, set_up_data
from protein_modifier.backend.protein_math import calculate_distance
from protein_modifier.backend.utils import get_sasa_by_residue
from protein_modifier.backend.data_structures import Atom, Residue, Chain, Structure
from protein_modifier.backend.io import parse_cif, write_cif, parse_structure
from protein_modifier.data.lammps_params import AA3_TO_IDX, MASSES, CHARGES
from protein_modifier.backend.build_idr import build_n_term_idr, build_c_term_idr, build_loop

def assign_bead_type(structure: Structure, structure_file: str, probe_radius: float = 1.4) -> Structure:
    structure = get_sasa_by_residue(structure, structure_file, probe_radius)
    for chain_id in structure.chains:
        for res_id in structure.chains[chain_id].residues:
            res_obj = structure.chains[chain_id].residues[res_id]
            aa_name = res_obj.name
            try:
                bead_type = AA3_TO_IDX[aa_name]
            except KeyError:
                raise ValueError(f"Unknown amino acid type '{aa_name}' in chain {chain_id} residue {res_id}")
            if res_obj.solvent_accessibility is None:
                raise ValueError(f"Residue {res_id} in chain {chain_id} does not have solvent accessibility assigned.")
            if res_obj.solvent_accessibility == 1:
                bead_id = bead_type + 20
            else:
                bead_id = bead_type
            res_obj.assign_bead_type(bead_id)
    return structure    

def generate_connect_lines(structure: Structure, bond_type: int = 1, warn_by_dist: bool = True,
                           dist_thresh: float = 6) -> list[str]:
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
                logger.warning(f"Distance between residue {res1} and {res2} in chain {chain} is {dist:.2f} Angstroms, which exceeds the threshold of {dist_thresh} Angstroms. This may indicate a problem with the structure or the assigned bead types.")
            # calculate distance between the
            bonds.append(f"{bond_num} {bond_type} {atom1_id} {atom2_id}")
            bond_num += 1
    return bonds

def write_seq_dat(structure_file_path: str, output_path: str, boxdims: float = 800,
                  num_atom_types: int = 75, num_bond_types: int = 1,
                  masses: list[float] | None = None, charges: dict[str, float] | None = None) -> None:
    """
    Generate a LAMMPS .dat data file from a coarse-grained structure.

    Parameters
    ----------
    structure_file_path : str
        Path to the input structure file (.cif or .pdb).
    output_path : str
        Path to write the LAMMPS data file.
    boxdims : float
        Simulation box dimensions (cubic, in angstroms). Default 800.
    num_atom_types : int
        Number of atom types in the LAMMPS file. Default 75.
    num_bond_types : int
        Number of bond types. Default 1.
    masses : list or None
        Custom mass list (length must match num_atom_types). If None, uses default MASSES.
    charges : dict or None
        Custom charge dict mapping 3-letter AA name to charge. If None, uses default CHARGES.
    """
    if masses is None:
        masses = MASSES
    if charges is None:
        charges = CHARGES
    structure = Structure.from_dict(parse_structure(structure_file_path))
    # center structure
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
    output_str += f"{num_atom_types} atom types\n"
    output_str += f"{num_bond_types} bond types\n\n"
    output_str += f"0.0 {boxdims}   xlo xhi\n"
    output_str += f"0.0 {boxdims}   ylo yhi\n"
    output_str += f"0.0 {boxdims}   zlo zhi\n\n"
    output_str += "Masses\n\n"
    for i in range(1, num_atom_types + 1):
        mass = masses[i-1]
        output_str += f"   {i} {mass:.6f}\n"
    output_str += "\nAtoms\n\n"
    atom_id = 1
    for chain in structure.chains:
        for residue in structure.chains[chain].residues:
            bead_type = structure.chains[chain].residues[residue].bead_type
            charge = charges[structure.chains[chain].residues[residue].name]
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

def find_string_indices_for_infile(structure: Structure, target_string: str) -> list[list[int]]:
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

def identify_gaps(data):
    """
    Identifies the start and end values of gaps in a sequence of integers.
    
    Args:
        data (list): A sorted list of integers.
        
    Returns:
        list of tuples: Each tuple contains (gap_start, gap_end).
    """
    if not data or len(data) < 2:
        return []

    gaps = []
    
    # Iterate through the list, comparing current element to the next
    for i in range(len(data) - 1):
        current_val = data[i]
        next_val = data[i + 1]
        
        # A gap exists if the difference is greater than 1
        if next_val - current_val > 1:
            # The gap begins at the integer immediately following current_val
            # and ends at the integer immediately preceding next_val
            gap_start = current_val + 1
            gap_end = next_val - 1
            gaps.append((gap_start, gap_end))
            
    return gaps

def identify_missing_ranges(data, min_val, max_val):
    """
    Identifies ranges of integers missing from the input list within 
    the inclusive bounds of [min_val, max_val].
    
    Args:
        data (list): A list of integers (assumed sorted).
        min_val (int): The start of the range to check.
        max_val (int): The end of the range to check.
        
    Returns:
        list of tuples: (gap_start, gap_end) for all missing segments.
    """
    # 1. Filter and sort data to ensure we only process relevant values within bounds
    # Using a set for filtering then sorting is efficient for uniqueness and order
    relevant_data = sorted([x for x in data if min_val <= x <= max_val])
    
    missing_ranges = []
    current_search = min_val

    # 2. Iterate through the relevant data to find gaps
    for val in relevant_data:
        # If the current value in our list is greater than the number 
        # we are looking for, a gap exists from current_search to val - 1
        if val > current_search:
            missing_ranges.append((current_search, val - 1))
        
        # Advance the search pointer to the number immediately after the current value
        current_search = val + 1

    # 3. Check for a trailing gap after the last element in relevant_data
    if current_search <= max_val:
        missing_ranges.append((current_search, max_val))

    return missing_ranges


def get_lammps_group_numbers(base_structure_path: str, json_input_path: str) -> str:
    """
    Function to get the numbers for the lammps groups that let us specify which
    regions to freeze and which to simulate. This function takes in the .json file that we 
    use for building the structure as well as the structure to identify which regions of 
    the structure are rebuilt. This assumes that the rebuilt regions are the ones that we
    want to simulate, and the non-rebuilt regions are the ones we want to freeze. This function
    only returns the group numbers as formatted for the lammps input file. It does not write the file. 

    Info on the format:
    1. Numbers are inclusive. 
    2. Numbers are 1 indexed.
    3. Numbers specified are the regions that are held constant (frozen).
    4. Each range that is held constant has the start number and end number separated by a colon
    5. Groups of numbers are separated by a space.

    Parameters
    ----------
    base_structure_path : str
        Path to the base structure used to generate the final structure. 
    json_input_path : str
        Path to the .json file that specifies how the structure is built. This is used to
        identify which regions of the structure are rebuilt and which are not.

    Returns    
    -------
    str
        A string formatted for the lammps input file that specifies the group numbers of the frozen regions.

    """

    build_data = set_up_data(read_build_file(json_input_path))

    structure_dict = parse_structure(base_structure_path)
    
    # wrangle data
    chain_sequences = {}
    for chain_info in build_data['chains_to_modify']:
        chain_id = chain_info['chain_id']
        sequence = chain_info['sequence']
        chain_sequences[chain_id] = sequence
    
    # now identify missing residues.
    missing_residue_dict = get_missing_residues_by_number(base_structure_path, chain_sequences)    

   # new need to set up build approaches. 
    build_instructions={}
    for chain_id in missing_residue_dict:
        build_instructions[chain_id] = {}
        chain_indices = missing_residue_dict[chain_id].keys()
        missing_chains = [i for i in chain_indices if missing_residue_dict[chain_id][i]['status'] == 'missing']
        if len(missing_chains)==0:
            continue
        else:
            for chain in missing_chains:
                # get indices for missing residues in this chain
                indices = missing_residue_dict[chain_id][chain]['index']
                # now change to amino acid numbers (not zero indexed)
                amino_acid_numbers = [int(i) + 1 for i in range(indices[0], indices[-1])]
                
                if len(amino_acid_numbers) == 0:
                    continue
                if len(amino_acid_numbers) != len(missing_residue_dict[chain_id][chain]['sequence']):
                    raise ValueError(f"Length of amino acid numbers does not match length of sequence for chain {chain} in chain_id {chain_id}")
                build_instructions[chain_id][chain] = {'sequence': missing_residue_dict[chain_id][chain]['sequence'],
                                                       'aa_nums': amino_acid_numbers,
                                                       'first_connecting_res': amino_acid_numbers[0] - 1,
                                                       'last_connecting_res': amino_acid_numbers[-1] + 1,
                                                       'build_type': None}
                if chain == 0:
                    build_instructions[chain_id][chain]['build_type'] = 'n_term'
                elif chain == max(chain_indices):
                    build_instructions[chain_id][chain]['build_type'] = 'c_term'
                else:
                    build_instructions[chain_id][chain]['build_type'] = 'loop'    
    
        # build the current structure
        current_structure = Structure.from_dict(structure_dict)
        if not current_structure.is_coarse_grained():
            current_structure = current_structure.coarse_grain()
            
        build_report=""
        
        # 5. Build missing residues
        for chain_id in build_instructions:
            for chain in build_instructions[chain_id]:
                instruction = build_instructions[chain_id][chain]
                if instruction['build_type'] == 'n_term':
                    current_structure = build_n_term_idr(target_structure=current_structure, 
                                                      chain_id=chain_id, 
                                                      new_idr_amino_acids = build_instructions[chain_id][chain]['sequence'],
                                                      stiffness_angle=build_data['stiffness_angle'],
                                                      bond_length=build_data['bond_length'],
                                                      clash_distance=0.1, # no clash checking for n-term since it's only attached on one side.
                                                      attempts=build_data['attempts'],
                                                      fake_build=True)
                    build_report += f"Built N-terminal IDR for chain {chain_id} with sequence {build_instructions[chain_id][chain]['sequence']}, residue numbers{build_instructions[chain_id][chain]['aa_nums']} \n"
                elif instruction['build_type'] == 'c_term':
                    current_structure = build_c_term_idr(target_structure=current_structure, 
                                                        chain_id=chain_id, 
                                                        new_idr_amino_acids = build_instructions[chain_id][chain]['sequence'],
                                                        stiffness_angle=build_data['stiffness_angle'],
                                                        bond_length=build_data['bond_length'],
                                                        clash_distance=0.1,
                                                        attempts=build_data['attempts'],
                                                        fake_build=True)
                    build_report += f"Built C-terminal IDR for chain {chain_id} with sequence {build_instructions[chain_id][chain]['sequence']}, residue numbers{build_instructions[chain_id][chain]['aa_nums']} \n"
                elif instruction['build_type'] == 'loop':
                    current_structure = build_loop(target_structure=current_structure, 
                                                   chain_id=chain_id, 
                                                   new_idr_amino_acids=build_instructions[chain_id][chain]['sequence'],
                                                   ind_of_first_connecting_atom=build_instructions[chain_id][chain]['first_connecting_res'],
                                                   ind_of_last_connecting_atom=build_instructions[chain_id][chain]['last_connecting_res'],
                                                   stiffness_angle=build_data['stiffness_angle'],
                                                   bond_length=build_data['bond_length'],
                                                   clash_distance=0.1,
                                                   attempts=build_data['attempts'],
                                                   fake_build=True)
                    build_report += f"Built loop IDR for chain {chain_id} with sequence {build_instructions[chain_id][chain]['sequence']}, residue numbers{build_instructions[chain_id][chain]['aa_nums']} \n"
                else:
                    raise ValueError(f"Unknown build instruction: {instruction}")
                
        # make sure input sequences match final sequences generated (full length)
        for n, chain_id in enumerate(build_data['chains_to_modify']):
            input_seq_id = build_data['chains_to_modify'][n]['chain_id']
            input_sequence = build_data['chains_to_modify'][n]['sequence']
            final_sequence = current_structure.chains[input_seq_id].get_amino_acid_sequence()
            if input_sequence != final_sequence:
                build_report += f"Warning: Final sequence for chain {chain_id} does not match input sequence. Input sequence: {input_sequence}, final sequence: {final_sequence}\n"
            else:
                build_report += f"Final sequence for chain {chain_id} matches input sequence.\n"
        
    # get indices of all built residues. 
    built_residues_info = current_structure.get_atom_index_of_built_residues()
    frozen_residues = identify_missing_ranges(built_residues_info, 1, len(current_structure.get_full_sequence()))
    # format for lammps input file
    frozen_residues_str = " ".join([f"{start}:{end}" for start, end in frozen_residues])
    return frozen_residues_str

def generate_lammps_infile(base_structure_path: str, json_input_path: str, output_path: str) -> None:
    # get the line we need
    lammps_group_numbers = get_lammps_group_numbers(base_structure_path, json_input_path)
    # get the random number
    random_number = random.randint(1000, 100000)
    # read the example file from the data folder
    with open(os.path.join(os.path.dirname(__file__), 'data', 'lammps_infile_base_v3.txt'), 'r') as f:
        lammps_input_str = f.read()
        # replace <INSERT_GROUP_ID_INDICES> with the lammps_group_numbers
        lammps_input_str = lammps_input_str.replace("<INSERT_GROUP_ID_INDICES>", lammps_group_numbers)
        # replace <INSERT_RANDOM_NUMBER> with the random number
        lammps_input_str = lammps_input_str.replace("<INSERT_RANDOM_NUMBER1>", str(random_number))
        lammps_input_str = lammps_input_str.replace("<INSERT_RANDOM_NUMBER2>", str(random_number))
    f.close()
    # write the new lammps input file    with open(output_path, 'w') as f:
    with open(output_path, 'w') as f:
        f.write(lammps_input_str)
    f.close()
    print(f'LAMMPS input file written to {output_path}')