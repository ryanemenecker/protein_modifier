"""
Docs for parser of build.json files.
"""
import os
import json
from protein_modifier.backend import default_parameters
from protein_modifier.data.amino_acids import AA_MAP_1_TO_3

def read_build_file(build_file_path: str) -> dict:
    """Parse a build.json file and return its contents as a dictionary."""
    with open(build_file_path, 'r') as f:
        build_data = json.load(f)
    return build_data

def set_up_data(build_data: dict) -> dict:
    """
    Process the raw build data and set up necessary parameters.
    """
    required_values = ['input_path', 'output_path']
    per_chain_params = ['sequence', 'chain_id']
    required_parameters =  ['bond_length', 'stiffness_angle', 'clash_distance', 'attempts']
    for val in required_values:
        if val not in build_data:
            raise ValueError(f"Missing required value: {val}")
    # Validate per-chain parameters
    seen_chain_ids = set()
    for chain in build_data.get('chains_to_modify', []):
        for param in per_chain_params:
            if param not in chain:
                raise ValueError(f"Missing required parameter '{param}' for chain: {chain}")
        # Check for duplicate chain IDs
        cid = chain['chain_id']
        if cid in seen_chain_ids:
            raise ValueError(f"Duplicate chain_id '{cid}' in chains_to_modify.")
        seen_chain_ids.add(cid)
        # Validate sequence contains only valid amino acid codes
        for aa in chain['sequence']:
            if aa not in AA_MAP_1_TO_3:
                raise ValueError(
                    f"Invalid amino acid code '{aa}' in sequence for chain '{cid}'. "
                    f"Valid codes: {', '.join(sorted(AA_MAP_1_TO_3.keys()))}"
                )
    for param in required_parameters:
        if param not in build_data:
            build_data[param] = getattr(default_parameters, param)
        else:
            # validate parameters
            if param == 'bond_length' and (not isinstance(build_data[param], (int, float)) or build_data[param] <= 0):
                raise ValueError(f"Invalid bond_length: {build_data[param]}. Must be a positive number.")
            if param == 'stiffness_angle' and (not isinstance(build_data[param], (int, float)) or not (0 < build_data[param] <= 180)):
                raise ValueError(f"Invalid stiffness_angle: {build_data[param]}. Must be a number between 0 and 180.")
            if param == 'clash_distance' and (not isinstance(build_data[param], (int, float)) or build_data[param] <= 0):
                raise ValueError(f"Invalid clash_distance: {build_data[param]}. Must be a positive number.")
            if param == 'attempts' and (not isinstance(build_data[param], int) or build_data[param] < 1):
                raise ValueError(f"Invalid attempts: {build_data[param]}. Must be a positive integer.")
    # Validate input file exists (checked last so structural errors are reported first)
    if not os.path.exists(build_data['input_path']):
        raise FileNotFoundError(f"Input structure file not found: {build_data['input_path']}")
    return build_data
