"""
Code for building loops - IDRs that are 
connected at both ends. 
"""

from sympy import centroid
from tqdm import tqdm
import numpy as np
from protein_modifier.backend.utils import calculate_distance
from protein_modifier.backend.data_structures import Atom, Structure, Chain, Residue
from protein_modifier.backend.io import parse_cif, write_cif
from protein_modifier.backend.modify_structure import get_neighbors_in_sphere, get_centroid, generate_sphere_points, extend_line_segment, get_non_clashing_coords, generate_next_calpha
from protein_modifier.data.amino_acids import AA_MAP_1_TO_3


def build_loop_coordinates(
        starting_coordinate: np.ndarray,
        ending_coordinate: np.ndarray,
        all_current_coordinates: np.ndarray,
        num_residues: int,
        start_index: int,
        bond_length: float = 3.8,
        clash_distance: float = 3.0,
        show_progress: bool = True):
    """
    Code to build the coordinates for a loop. Uses a simple
    reducing sphere size approach.

    Parameters:
    - starting_coordinate: (3,) array of the starting point (e.g. CA of last resolved residue)
    - ending_coordinate: (3,) array of the ending point (e.g. CA of first resolved residue)
    - all_current_coordinates: (N, 3) array of all existing atom coordinates to avoid clashes with
    - num_residues: How many residues to build in the IDR segment
    - start_index: The residue index to assign to the first built residue (e.g. if building between res 50 and 60, start_index would be 51)
    - bond_length: Distance to next atom (default 3.8 Angstroms for CA-CA)
    - clash_distance: Minimum allowed distance to existing atoms (default 3.0 Angstroms)
    - show_progress: Whether to display a progress bar (default True)

    Returns:
    - np.ndarray of (N, 3) array of new coordinates for the loop
    """
    # make list to hold new atoms.
    new_coords = {}
    all_coords=[]
    all_coords.extend(all_current_coordinates)

    # identify coordinates within 40 angstroms of the start and end. 
    starting_neighbors = get_neighbors_in_sphere(starting_coordinate, all_current_coordinates, radius=40)
    if len(starting_neighbors)>0:
        # get centroids
        start_centroid = get_centroid(starting_neighbors) if len(starting_neighbors) > 0 else None        
        if np.linalg.norm(start_centroid - starting_coordinate) > 1e-3:
            # extend line from centroid to connecting_atom_coords by bond_length
            first_pos = extend_line_segment(start_centroid, starting_coordinate, bond_length)
            # check for clashing.
            first_pos_candidates = get_non_clashing_coords(first_pos, all_current_coordinates, min_distance=clash_distance)
            if len(first_pos_candidates)==0:
                first_pos=None
            else:
                first_pos = first_pos_candidates[0]
        else:
            first_pos = None
    else:
        first_pos = None
    
    if first_pos is None:
        # if we couldn't find a good direction, just generate candidates in a sphere and pick one that doesn't clash.
        candidate_positions = generate_sphere_points(starting_coordinate, bond_length, num_points=100)
        candidate_positions = get_non_clashing_coords(candidate_positions, all_current_coordinates, min_distance=clash_distance)
        if len(candidate_positions)==0:
            raise ValueError("Could not find any non-clashing positions for the first residue.")
        first_pos = candidate_positions[0]
    all_coords.append(first_pos)
    new_coords[start_index] = first_pos

    # now same for the end. 
    ending_neighbors = get_neighbors_in_sphere(ending_coordinate, all_current_coordinates, radius=40)
    if len(ending_neighbors)>0:
        # get centroids
        end_centroid = get_centroid(ending_neighbors) if len(ending_neighbors) > 0 else None        
        if np.linalg.norm(end_centroid - ending_coordinate) > 1e-3:
            # extend line from centroid to connecting_atom_coords by bond_length
            last_pos = extend_line_segment(end_centroid, ending_coordinate, bond_length)
            # check for clashing.
            last_pos_candidates = get_non_clashing_coords(last_pos, all_current_coordinates, min_distance=clash_distance)
            if len(last_pos_candidates)==0:
                last_pos=None
            else:
                last_pos = last_pos_candidates[0]
        else:
            last_pos = None
    else:
        last_pos = None
    
    if last_pos is None:
        # if we couldn't find a good direction, just generate candidates in a sphere and pick one that doesn't clash.
        candidate_positions = generate_sphere_points(ending_coordinate, bond_length, num_points=100)
        candidate_positions = get_non_clashing_coords(candidate_positions, all_current_coordinates, min_distance=clash_distance)
        if len(candidate_positions)==0:
            raise ValueError("Could not find any non-clashing positions for the last residue.")
        last_pos = candidate_positions[0]
    all_coords.append(last_pos)
    new_coords[start_index+num_residues-1] = last_pos

    # now we want to connect from the first to the last. They should both now point away from the existing structure.
    # get the radius from the first pos to the last_pos
    start_radius = calculate_distance(first_pos, last_pos)
    # make sure is theoretically possible (dist must be less than length-2*bond_length)
    if start_radius > bond_length*(num_residues-1):
        raise ValueError("The distance between the first and last positions is too great to connect with the given number of residues and bond length.")
    
    # now we build a sphere of potential points from first_coord that are within a shrinking
    # radius forming a sphere from last_pos. We pick a random value from this list that is
    # not clashing.
    for i in tqdm(range(1, num_residues-1), desc="Building loop coordinates", disable=not show_progress):
        # calculate the current radius we want to be from the last_pos. This should shrink linearly from start_radius to bond_length.
        current_radius = start_radius - (start_radius - bond_length) * (i / (num_residues-1))
        # generate candidate positions in a sphere around the last position with this radius.
        candidate_positions = generate_sphere_points(first_pos, bond_length, num_points=100)
        # now find candidates within the sphere.
        candidate_positions = get_neighbors_in_sphere(last_pos, candidate_positions, radius=current_radius)
        # filter out candidates that clash with existing structure.
        candidate_positions = get_non_clashing_coords(candidate_positions, all_coords, min_distance=clash_distance)
        if len(candidate_positions)==0:
            raise ValueError(f"Could not find any non-clashing positions for residue {start_index+i}.")
        next_pos = candidate_positions[0]
        new_coords[start_index+i] = next_pos
        all_coords.append(next_pos)
        first_pos = next_pos
    
    # sort dictionary by smallest to largest key and return as array.
    sorted_coords = [new_coords[i] for i in sorted(new_coords.keys())]
    return np.array(sorted_coords)

