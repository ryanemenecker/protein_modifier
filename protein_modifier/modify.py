"""Provide the primary functions."""
import os

from protein_modifier.backend.build_idr import build_n_term_idr, build_c_term_idr, build_loop
from protein_modifier.backend.find_missing_res import get_missing_residues_by_number
from protein_modifier.backend.data_structures import Atom, Residue, Chain, Structure
from protein_modifier.backend.io import parse_cif, write_cif
from protein_modifier.backend import default_parameters
from protein_modifier.backend.parse_build_file import read_build_file, set_up_data

def modify_protein(build_file_path: str,
                   coarse_grain=True): 
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
    
    # 2. Parse the input structure
    structure_dict = parse_cif(build_data['input_path'])
    current_structure = Structure.from_dict(structure_dict)
    
    # 3. coarse grainify structure (if not already)
    if not current_structure.is_coarse_grained():
        current_structure = current_structure.coarse_grain()
    
    # 4. wrangle data
    chain_sequences = {}
    for chain_info in build_data['chains_to_modify']:
        chain_id = chain_info['chain_id']
        sequence = chain_info['sequence']
        chain_sequences[chain_id] = sequence
    
    # now identify missing residues.
    missing_residue_dict = get_missing_residues_by_number(build_data['input_path'], chain_sequences)

    # new need to set up build approaches. 
    build_report=""
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
                                                  clash_distance=build_data['clash_distance'])
                build_report += f"Built N-terminal IDR for chain {chain_id} with sequence {build_instructions[chain_id][chain]['sequence']}, residue numbers{build_instructions[chain_id][chain]['aa_nums']} \n"
            elif instruction['build_type'] == 'c_term':
                current_structure = build_c_term_idr(target_structure=current_structure, 
                                                    chain_id=chain_id, 
                                                    new_idr_amino_acids = build_instructions[chain_id][chain]['sequence'],
                                                    stiffness_angle=build_data['stiffness_angle'],
                                                    bond_length=build_data['bond_length'],
                                                    clash_distance=build_data['clash_distance'])
                build_report += f"Built C-terminal IDR for chain {chain_id} with sequence {build_instructions[chain_id][chain]['sequence']}, residue numbers{build_instructions[chain_id][chain]['aa_nums']} \n"
            elif instruction['build_type'] == 'loop':
                current_structure = build_loop(target_structure=current_structure, 
                                               chain_id=chain_id, 
                                               new_idr_amino_acids=build_instructions[chain_id][chain]['sequence'],
                                               ind_of_first_connecting_atom=build_instructions[chain_id][chain]['first_connecting_res'],
                                               ind_of_last_connecting_atom=build_instructions[chain_id][chain]['last_connecting_res'],
                                               stiffness_angle=build_data['stiffness_angle'],
                                               bond_length=build_data['bond_length'],
                                               clash_distance=build_data['clash_distance'])
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
    # write output
    print(f"Writing modified structure to {build_data['output_path']}...")
    write_cif(current_structure.to_dict(), build_data['output_path'])
    # write out report of what was done. 
    report_path = os.path.splitext(build_data['output_path'])[0] + "_build_report.txt"
    with open(report_path, 'w') as f:
        f.write(build_report)
    print(f"Build report written to {report_path}")
    print("Done.")


                    

# example usage

structure = Structure.from_dict(parse_cif('/Users/ryanemenecker/Desktop/simulations/lammps_complexes/actin_complex_round2_return_of_the_calcium/6KN8-assembly1.cif'))
# coarse grain
structure = structure.coarse_grain()
# write structure to cif.
write_cif(structure.to_dict(), '/Users/ryanemenecker/Desktop/simulations/lammps_complexes/actin_complex_round2_return_of_the_calcium/6KN8-assembly-cg.cif')   

json_loc = "/Users/ryanemenecker/Desktop/simulations/lammps_complexes/actin_complex_round2_return_of_the_calcium/build_6kn8.json"
modify_protein(json_loc)


