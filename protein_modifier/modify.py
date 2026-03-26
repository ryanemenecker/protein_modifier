"""Provide the primary functions."""
from __future__ import annotations

import os
import logging

logger = logging.getLogger(__name__)

from protein_modifier.backend.build_idr import build_n_term_idr, build_c_term_idr, build_loop
from protein_modifier.backend.find_missing_res import get_missing_residues_by_number
from protein_modifier.backend.data_structures import Atom, Residue, Chain, Structure
from protein_modifier.backend.io import parse_cif, write_cif, write_pdb, parse_pdb, parse_structure
from protein_modifier.backend import default_parameters
from protein_modifier.backend.parse_build_file import read_build_file, set_up_data

def modify_protein(build_file_path: str,
                   coarse_grain: bool = True) -> None: 
    """
    Main function to modify a protein structure based on a build file.

    Parameters:
    - build_file_path: Path to the JSON file containing build instructions.
    - coarse_grain: If True, only build C-alpha atoms. Currently only supports coarse-grained building.
    """
    if coarse_grain is not True:
        raise NotImplementedError("Fine-grained building (with side chains) is not yet implemented.")
    
    # 1. Read and set up data from the build file
    build_data = read_build_file(build_file_path)
    build_data = set_up_data(build_data)
    
    # 2. Parse the input structure (auto-detect .cif or .pdb)
    input_path = build_data['input_path']
    structure_dict = parse_structure(input_path)
    
    # 4. wrangle data
    chain_sequences = {}
    for chain_info in build_data['chains_to_modify']:
        chain_id = chain_info['chain_id']
        sequence = chain_info['sequence']
        chain_sequences[chain_id] = sequence
    
    # now identify missing residues.
    missing_residue_dict = get_missing_residues_by_number(input_path, chain_sequences)

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

    num_replicates = build_data['replicates']
    
    for replicate_idx in range(1, num_replicates + 1):
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
                                                      clash_distance=build_data['clash_distance'],
                                                      attempts=build_data['attempts'])
                    build_report += f"Built N-terminal IDR for chain {chain_id} with sequence {build_instructions[chain_id][chain]['sequence']}, residue numbers{build_instructions[chain_id][chain]['aa_nums']} \n"
                elif instruction['build_type'] == 'c_term':
                    current_structure = build_c_term_idr(target_structure=current_structure, 
                                                        chain_id=chain_id, 
                                                        new_idr_amino_acids = build_instructions[chain_id][chain]['sequence'],
                                                        stiffness_angle=build_data['stiffness_angle'],
                                                        bond_length=build_data['bond_length'],
                                                        clash_distance=build_data['clash_distance'],
                                                        attempts=build_data['attempts'])
                    build_report += f"Built C-terminal IDR for chain {chain_id} with sequence {build_instructions[chain_id][chain]['sequence']}, residue numbers{build_instructions[chain_id][chain]['aa_nums']} \n"
                elif instruction['build_type'] == 'loop':
                    current_structure = build_loop(target_structure=current_structure, 
                                                   chain_id=chain_id, 
                                                   new_idr_amino_acids=build_instructions[chain_id][chain]['sequence'],
                                                   ind_of_first_connecting_atom=build_instructions[chain_id][chain]['first_connecting_res'],
                                                   ind_of_last_connecting_atom=build_instructions[chain_id][chain]['last_connecting_res'],
                                                   stiffness_angle=build_data['stiffness_angle'],
                                                   bond_length=build_data['bond_length'],
                                                   clash_distance=build_data['clash_distance'],
                                                   attempts=build_data['attempts'])
                    build_report += f"Built loop IDR for chain {chain_id} with sequence {build_instructions[chain_id][chain]['sequence']}, residue numbers{build_instructions[chain_id][chain]['aa_nums']} \n"
                else:
                    raise ValueError(f"Unknown build instruction: {instruction}")
        
        # check for final clashing
        is_clashing = not current_structure.verify_non_clashing(min_distance=build_data['clash_distance'])
        if is_clashing:
            build_report += "Warning: Final structure has clashing atoms based on the specified clash distance.\n"
        else:
            build_report += "No clashing atoms detected in the final structure based on the specified clash distance.\n"
        
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
        # add this info to the build report.
        build_report += "\n" + str(built_residues_info) + "\n"
        
        # determine output directory based on replicates
        output_path = build_data['output_path']
        output_dir = os.path.dirname(output_path)
        output_filename = os.path.basename(output_path)
        if num_replicates > 1:
            rep_dir = os.path.join(output_dir, f"replicate_{replicate_idx}") if output_dir else f"replicate_{replicate_idx}"
            os.makedirs(rep_dir, exist_ok=True)
            current_output_path = os.path.join(rep_dir, output_filename)
        else:
            current_output_path = output_path
            
        output_ext = os.path.splitext(current_output_path)[1].lower()
        logger.info(f"Writing modified structure to {current_output_path}...")
        if output_ext == '.pdb':
            write_pdb(current_structure.to_dict(), current_output_path)
        else:
            write_cif(current_structure.to_dict(), current_output_path)
            
        # write out report of what was done. 
        report_path = os.path.splitext(current_output_path)[0] + "_build_report.txt"
        with open(report_path, 'w') as f:
            f.write(build_report)
        logger.info(f"Build report written to {report_path}")
        
    logger.info("Done.")

