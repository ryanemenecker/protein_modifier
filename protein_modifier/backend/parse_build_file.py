"""
Docs for parser of build.json files.
"""
import json
from protein_modifier.backend import default_parameters

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
    required_parameters =  ['bond_length', 'stiffness_angle', 'clash_distance']
    for val in required_values:
        if val not in build_data:
            raise ValueError(f"Missing required value: {val}")
    for chain in build_data.get('chains_to_modify'):
        for param in per_chain_params:
            if param not in chain:
                raise ValueError(f"Missing required parameter '{param}' for chain: {chain}")
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
    return build_data
