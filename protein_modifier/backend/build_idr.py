"""
Docstring for protein_modifier.backend.build_idr
"""
from __future__ import annotations

import logging
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)
from protein_modifier.backend.data_structures import Atom, Structure, Chain, Residue
from protein_modifier.backend.io import parse_cif, write_cif
from protein_modifier.backend.modify_structure import get_neighbors_in_sphere, get_centroid, generate_sphere_points, extend_line_segment, get_non_clashing_coords, generate_next_calpha
from protein_modifier.data.amino_acids import AA_MAP_1_TO_3
from protein_modifier.backend.protein_math import calculate_distance, find_furthest_coordinate, find_points_within_sphere


def _candidate_clearances(candidates: np.ndarray, obstacles: np.ndarray) -> np.ndarray:
    if len(candidates) == 0:
        return np.array([])
    distances = np.linalg.norm(candidates[:, np.newaxis, :] - obstacles[np.newaxis, :, :], axis=2)
    return distances.min(axis=1)


def _generate_sphere_intersection_points(
        center_a: np.ndarray,
        center_b: np.ndarray,
        radius_a: float,
        radius_b: float,
        num_points: int = 720) -> np.ndarray:
    center_a = np.asarray(center_a, dtype=float)
    center_b = np.asarray(center_b, dtype=float)
    distance = np.linalg.norm(center_b - center_a)
    tolerance = 1e-6

    if distance < tolerance:
        return np.empty((0, 3))
    if distance > (radius_a + radius_b + tolerance):
        return np.empty((0, 3))
    if distance < (abs(radius_a - radius_b) - tolerance):
        return np.empty((0, 3))

    distance = min(distance, radius_a + radius_b)

    axis = (center_b - center_a) / distance
    circle_center_distance = (radius_a ** 2 - radius_b ** 2 + distance ** 2) / (2 * distance)
    circle_radius_sq = radius_a ** 2 - circle_center_distance ** 2
    if circle_radius_sq < 0:
        if circle_radius_sq > -1e-6:
            circle_radius_sq = 0.0
        else:
            return np.empty((0, 3))

    circle_center = center_a + axis * circle_center_distance
    circle_radius = np.sqrt(circle_radius_sq)

    reference = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    basis_u = np.cross(axis, reference)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(axis, basis_u)

    angles = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    return np.array([
        circle_center + circle_radius * (np.cos(angle) * basis_u + np.sin(angle) * basis_v)
        for angle in angles
    ])


def _generate_final_loop_candidates(
        current_coordinate: np.ndarray,
        ending_coordinate: np.ndarray,
        current_coords: np.ndarray,
        bond_length: float,
        clash_distance: float,
        num_points: int = 720) -> np.ndarray:
    candidates = _generate_sphere_intersection_points(
        current_coordinate,
        ending_coordinate,
        bond_length,
        bond_length,
        num_points=num_points,
    )
    if len(candidates) == 0:
        return candidates
    return get_non_clashing_coords(candidates, current_coords, min_distance=clash_distance)


def _score_loop_candidates(
        candidates: np.ndarray,
        ending_coordinate: np.ndarray,
        current_coords: np.ndarray,
        desired_distance: float) -> np.ndarray:
    end_distances = np.linalg.norm(candidates - ending_coordinate, axis=1)
    clearances = _candidate_clearances(candidates, current_coords)
    return (0.25 * clearances) - np.abs(end_distances - desired_distance)


def _build_loop_coordinates_flexible(
        starting_coordinate: np.ndarray,
        ending_coordinate: np.ndarray,
        all_current_coordinates: np.ndarray,
        num_residues: int,
        start_index: int,
        bond_length: float = 3.8,
        clash_distance: float = 3.0,
        show_progress: bool = True,
        sphere_points_per_step: int = 2500,
        branch_factor: int = 12,
        beam_width: int = 24) -> np.ndarray:
    beam = [([], np.asarray(starting_coordinate, dtype=float), np.asarray(all_current_coordinates, dtype=float))]
    print(num_residues)
    for i in tqdm(range(num_residues), disable=not show_progress):
        remaining_bonds = num_residues - i
        next_states = []
        for path, current_coordinate, current_coords in beam:
            if remaining_bonds == 1:

                candidates = _generate_final_loop_candidates(
                    current_coordinate,
                    ending_coordinate,
                    current_coords,
                    bond_length,
                    clash_distance,
                )
                desired_distance = bond_length
            elif remaining_bonds == 2:

                print('GENERATING SPHERE INTERSECTION POINT')
                print(current_coordinate)
                print(ending_coordinate)
                print(np.linalg.norm(np.array(current_coordinate)-np.array(ending_coordinate)))

                candidates = _generate_sphere_intersection_points(
                    current_coordinate,
                    ending_coordinate,
                    bond_length,
                    2 * bond_length,
                )
                candidates = get_non_clashing_coords(candidates, current_coords, min_distance=clash_distance)
                if len(candidates) == 0:
                    continue
                desired_distance = 2 * bond_length
            else:
                sphere_points = generate_sphere_points(current_coordinate, radius=bond_length, num_points=sphere_points_per_step)
                candidates = get_non_clashing_coords(sphere_points, current_coords, min_distance=clash_distance)
                if len(candidates) == 0:
                    continue

                max_reachable_distance = remaining_bonds * bond_length + 1e-6
                end_distances = np.linalg.norm(candidates - ending_coordinate, axis=1)
                candidates = candidates[end_distances <= max_reachable_distance]
                if len(candidates) == 0:
                    candidates = _generate_sphere_intersection_points(
                        current_coordinate,
                        ending_coordinate,
                        bond_length,
                        max_reachable_distance,
                    )
                    if len(candidates) == 0:
                        continue
                    candidates = get_non_clashing_coords(candidates, current_coords, min_distance=clash_distance)
                    if len(candidates) == 0:
                        continue

                current_end_distance = np.linalg.norm(current_coordinate - ending_coordinate)
                desired_distance = max(
                    bond_length,
                    min(current_end_distance - bond_length, remaining_bonds * bond_length)
                )

            candidate_scores = _score_loop_candidates(candidates, ending_coordinate, current_coords, desired_distance)
            top_indices = np.argsort(candidate_scores)[-branch_factor:][::-1]

            for idx in top_indices:
                candidate = candidates[idx]
                candidate_path = path + [candidate]
                candidate_coords = np.vstack((current_coords, candidate))
                remaining_after_candidate = remaining_bonds - 1
                final_distance = np.linalg.norm(candidate - ending_coordinate)
                total_score = candidate_scores[idx] - (0.1 * remaining_after_candidate * final_distance)
                next_states.append((total_score, candidate_path, candidate, candidate_coords))

        if not next_states:
            raise ValueError(f"Could not find a reachable non-clashing candidate for loop residue {start_index + i}")

        next_states.sort(key=lambda item: item[0], reverse=True)
        beam = [(path, current_coordinate, current_coords)
                for _, path, current_coordinate, current_coords in next_states[:beam_width]]

    return np.array(beam[0][0])


def _build_loop_coordinates_greedy(
        starting_coordinate: np.ndarray,
        ending_coordinate: np.ndarray,
        all_current_coordinates: np.ndarray,
        num_residues: int,
        start_index: int,
        bond_length: float = 3.8,
        clash_distance: float = 3.0,
        show_progress: bool = True) -> np.ndarray:
    current_coords = all_current_coordinates.copy()
    new_coords = []

    for i in tqdm(range(num_residues), disable=not show_progress):
        residues_left_to_place = num_residues - i
        if residues_left_to_place == 1:
            candidates = _generate_final_loop_candidates(
                starting_coordinate,
                ending_coordinate,
                current_coords,
                bond_length,
                clash_distance,
            )
            if len(candidates) == 0:
                raise ValueError(f"Could not find feasible non-clashing candidates for loop residue {start_index + i}")
            final_coord = candidates[0]
        elif residues_left_to_place == 2:
            candidates = _generate_sphere_intersection_points(
                starting_coordinate,
                ending_coordinate,
                bond_length,
                2 * bond_length,
            )
            candidates = get_non_clashing_coords(candidates, current_coords, min_distance=clash_distance)
            if len(candidates) == 0:
                raise ValueError(f"Could not find feasible non-clashing candidates for loop residue {start_index + i}")
            final_coord = candidates[0]
        else:
            sphere_points = generate_sphere_points(starting_coordinate, radius=bond_length, num_points=5000)
            candidates = get_non_clashing_coords(sphere_points, current_coords, min_distance=clash_distance)
            if len(candidates) == 0:
                raise ValueError(f"Could not find non-clashing candidates for loop residue {start_index + i}")

            max_reachable_distance = residues_left_to_place * bond_length + 1e-6
            candidates = find_points_within_sphere(candidates, ending_coordinate, max_reachable_distance)
            if len(candidates) == 0:
                candidates = _generate_sphere_intersection_points(
                    starting_coordinate,
                    ending_coordinate,
                    bond_length,
                    max_reachable_distance,
                )
                candidates = get_non_clashing_coords(candidates, current_coords, min_distance=clash_distance)
                if len(candidates) == 0:
                    raise ValueError(f"No candidates remain within a reachable end distance for loop residue {start_index + i}")

            dists = np.linalg.norm(candidates - ending_coordinate, axis=1)
            best_index = np.argmax(_score_loop_candidates(candidates, ending_coordinate, current_coords, max_reachable_distance))
            final_coord = candidates[best_index]

        new_coords.append(final_coord)
        current_coords = np.vstack((current_coords, final_coord))
        starting_coordinate = final_coord

    return np.array(new_coords)

def build_idr_coordinates(
        connecting_atom_coords: np.ndarray,
        num_residues: int,
        current_coordinates: np.ndarray,
        bond_length: float = 3.8,
        stiffness_angle: float = 120,
        show_progress: bool = True,
        clash_distance: float = 3.0,
        fake_build: bool = False
    ) -> list[np.ndarray]:
    """Builds a simple, random(ish) IDR segment of a chain.
    Parameters:
    - connecting_atom_coords: (x,y,z) of the atom to connect to (e.g. CA of first resolved residue)
    - num_residues: How many residues to build in the IDR segment
    - current_coordinates: List of (x,y,z) of currently resolved structure (used for collision checking)
    - bond_length: Distance to next atom (default 3.8 Angstroms for CA-CA)
    - stiffness_angle: The bond angle in degrees (angle between p_prev-p_pprev and new_vec).
                       180 = perfectly straight chain.
                       90 = sharp turn.
    - show_progress: Whether to show a progress bar (useful for long IDRs)
    - clash_distance: Minimum distance to avoid clashes with existing atoms
    - fake_build: If True, the function will simulate building without checking for clashes or generating realistic coordinates. This is useful for testing the integration of the build process without relying on the geometry functions.
    Returns:
    - List of new atom dicts with keys: x, y, z
    """
    # list to hold new atoms.
    new_atoms = []
    if fake_build:
        # Just generate points in a line for testing purposes.
        for i in range(num_residues):
            new_atoms.append(connecting_atom_coords + np.array([bond_length * (i+1), 0, 0]))
        return new_atoms
    else:
        # identify coordinates within 20 angstroms of the connecting
        # atom coordinate so we can get a directional vector for the first step.
        neighbors = get_neighbors_in_sphere(connecting_atom_coords, current_coordinates, radius=40)
        # get centroid
        use_random = True
        if len(neighbors) > 0:
            centroid = get_centroid(neighbors)
            if np.linalg.norm(centroid - connecting_atom_coords) > 1e-3:
                # extend line from centroid to connecting_atom_coords by bond_length
                first_pos = extend_line_segment(centroid, connecting_atom_coords, bond_length)
                use_random = False

        if use_random:
            # If no neighbors, just pick a random point at the correct distance.
            # This is a fallback and may lead to worse initial geometry.
            random_dir = np.random.randn(3)
            random_dir /= np.linalg.norm(random_dir)
            first_pos = connecting_atom_coords + random_dir * bond_length
        # Ensure the first position doesn't clash with existing structure
        candidates = get_non_clashing_coords(first_pos, current_coordinates, min_distance=clash_distance)
        
        if len(candidates) == 0:
            # generate points in sphere as a backup.
            sphere_points = generate_sphere_points(connecting_atom_coords, radius=bond_length, num_points=500)
            candidates = get_non_clashing_coords(sphere_points, current_coordinates, min_distance=clash_distance)
            if len(candidates) == 0:
                raise ValueError("Could not find a non-clashing position for the first IDR atom.")
                
        new_atoms.append(candidates[0])
        
        # Now iteratively build the rest of the chain
        for i in tqdm(range(1, num_residues), disable=not show_progress):
            next_pos = generate_next_calpha(new_atoms[-1], new_atoms[-2] if i > 1 else connecting_atom_coords, bond_length, stiffness_angle)
            candidates = get_non_clashing_coords(next_pos, current_coordinates, min_distance=clash_distance)
            if len(candidates) > 0:
                candidates = get_non_clashing_coords(candidates, np.array(new_atoms), min_distance=clash_distance)
            
            if len(candidates) == 0:
                # If the generated position clashes, try random points in a sphere around the last position.
                sphere_points = generate_sphere_points(new_atoms[-1], radius=bond_length, num_points=100)
                candidates = get_non_clashing_coords(sphere_points, current_coordinates, min_distance=clash_distance)
                candidates = get_non_clashing_coords(candidates, np.array(new_atoms), min_distance=clash_distance)
                if len(candidates) == 0:
                    raise ValueError(f"Could not find a non-clashing position for IDR atom {i}.")
                    
            new_atoms.append(candidates[0])
        return new_atoms


def build_loop_coordinates(
        starting_coordinate: np.ndarray,
        ending_coordinate: np.ndarray,
        all_current_coordinates: np.ndarray,
        num_residues: int,
        start_index: int,
        bond_length: float = 3.8,
        clash_distance: float = 3.0,
        show_progress: bool = True,
        fake_build: bool = False) -> np.ndarray:
    """
    Build coordinates for a loop between two anchors.

    The primary path uses a greedy reachability-guided search; if that fails,
    a more flexible fallback explores exact bridge geometry while preserving
    clash checks and bond lengths.

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
    if fake_build:
        # Just generate points in a line for testing purposes.
        return np.array([
            starting_coordinate + np.array([bond_length * (i + 1), 0, 0])
            for i in range(num_residues)
        ])
    try:
        return _build_loop_coordinates_greedy(
            starting_coordinate=starting_coordinate,
            ending_coordinate=ending_coordinate,
            all_current_coordinates=all_current_coordinates,
            num_residues=num_residues,
            start_index=start_index,
            bond_length=bond_length,
            clash_distance=clash_distance,
            show_progress=show_progress,
        )
    except ValueError as greedy_error:
        logger.warning(
            "Greedy loop builder failed for residues starting at %s: %s. Falling back to flexible loop closure.",
            start_index,
            greedy_error,
        )
        try:
            return _build_loop_coordinates_flexible(
                starting_coordinate=starting_coordinate,
                ending_coordinate=ending_coordinate,
                all_current_coordinates=all_current_coordinates,
                num_residues=num_residues,
                start_index=start_index,
                bond_length=bond_length,
                clash_distance=clash_distance,
                show_progress=show_progress,
            )
        except ValueError as flexible_error:
            logger.warning(
                "Flexible loop closure failed for residues starting at %s: %s.",
                start_index,
                flexible_error,
            )
            raise ValueError(
                f"Flexible loop closure failed for residues starting at {start_index}: {flexible_error}"
            ) from flexible_error

def add_atoms_to_structure(structure: Structure, chain_id: str, new_atoms: list | np.ndarray,
                           residue_names: list[str], atom_names: list[str],
                           start_ind: int = 1) -> Structure:
    """Helper function to add new atoms to the structure in the correct format."""
    chain = structure.chains.get(chain_id)
    if not chain:
        chain = Chain(chain_id)
        structure.chains[chain_id] = chain
    
    for i, (atom_coords, res_name, atom_name) in enumerate(zip(new_atoms, residue_names, atom_names)):
        res_id = start_ind + i
        residue = chain.get_or_create_residue(res_id, res_name)
        atom_dict = {
            'label_comp_id': res_name,
            'label_asym_id': chain_id,
            'label_seq_id': res_id,
            'Cartn_x': atom_coords[0],
            'Cartn_y': atom_coords[1],
            'Cartn_z': atom_coords[2],
            'name': atom_name
        }
        residue.add_atom(Atom(atom_dict), set_was_built=True)
    return structure

def build_c_term_idr(
        target_structure: Structure,
        chain_id: str,
        new_idr_amino_acids: str,
        connecting_atom_name: str = 'CA',
        start_ind: int = None,
        show_progress: bool = True,
        stiffness_angle: float = 120,
        bond_length: float = 3.8,
        clash_distance: float = 3.0,
        attempts: int = 5,
        fake_build: bool = False
    ) -> Structure:
    """
    Docstring for build_c_term_idr
    
    parameters
    ----------
    - target_structure: Structure object representing the protein structure to modify.
    - chain_id: The ID of the chain to which the C-terminal IDR should be added.
    - new_idr_amino_acids: List of 1-letter amino acid codes for the new IDR segment "ACDEF"
    - connecting_atom_name: The name of the atom in the first resolved residue to connect to (default 'CA').
    - start_ind: The starting residue index for the new IDR segment (default 1).
    - show_progress: Whether to display a progress bar during IDR construction (default True).
    - stiffness_angle: The bond angle in degrees for the random walk (default 120, where 180 is straight and 90 is a sharp turn).
    - bond_length: The distance between consecutive C-alpha atoms (default 3.8 Angstroms).
    - clash_distance: Minimum distance to avoid clashes with existing atoms (default 3.0 Angstroms).
    - attempts: Number of attempts to build the IDR if clashes are detected (default 5). If all attempts fail, an error is raised.
    - fake_build: If True, the function will simulate building without checking for clashes or generating realistic coordinates. This is useful for testing the integration of the build process without relying on the geometry functions.
    returns
    -------
    - modified_structure: A new Structure object with the C-terminal IDR added.
    """
    all_atoms = target_structure.get_coords()
    
    # Sort residues by ID to correctly identify the sequence C-terminus
    # (Handling integer IDs vs string IDs)
    chain_residues = target_structure.chains[chain_id].residues
    def res_key(k):
        try: return int(k)
        except ValueError: return -999999
        
    sorted_keys = sorted(chain_residues.keys(), key=res_key)
    if not sorted_keys:
        raise ValueError(f"Chain {chain_id} has no residues.")
        
    last_ca_ind = sorted_keys[-1]
    
    if start_ind is None:
        start_ind = int(last_ca_ind) + 1

    last_ca_coord = target_structure.chains[chain_id].residues[last_ca_ind][connecting_atom_name]
    last_ca_coordinates = np.array([last_ca_coord.x, last_ca_coord.y, last_ca_coord.z])
    for i in range(attempts):
        try:
            new_idr_atoms = build_idr_coordinates(
                connecting_atom_coords=last_ca_coordinates,
                num_residues=len(new_idr_amino_acids),
                current_coordinates=all_atoms,
                bond_length=bond_length,
                stiffness_angle=stiffness_angle,
                show_progress=show_progress,
                clash_distance=clash_distance,
                fake_build=fake_build
            )
            break
        except Exception as e:
            if i == attempts - 1:
                raise ValueError(
                    f"Failed to build C-terminal IDR for chain '{chain_id}' "
                    f"({len(new_idr_amino_acids)} residues) after {attempts} attempts. "
                    f"Last error: {e}. "
                    f"Try reducing stiffness_angle (current: {stiffness_angle}) or "
                    f"clash_distance (current: {clash_distance})."
                )
            else:
                logger.warning(f"Attempt {i+1}/{attempts} failed for C-terminal IDR on chain '{chain_id}': {e}. Retrying...")
                continue
    res_names = [AA_MAP_1_TO_3[res] for res in new_idr_amino_acids]
    atom_names = ['CA'] * len(res_names)
    updated_struct = add_atoms_to_structure(target_structure, chain_id, new_idr_atoms, res_names, atom_names, start_ind=start_ind)
    return updated_struct

def build_n_term_idr(
        target_structure: Structure,
        chain_id: str,
        new_idr_amino_acids: str,
        connecting_atom_name: str = 'CA',
        start_ind: int = 1,
        show_progress: bool = True,
        stiffness_angle: float = 120,
        bond_length: float = 3.8,
        clash_distance: float = 3.0,
        attempts: int = 5,
        fake_build: bool = False

    ) -> Structure:
    """
    Docstring for build_n_term_idr
    
    parameters
    ----------
    - target_structure: Structure object representing the protein structure to modify.
    - chain_id: The ID of the chain to which the N-terminal IDR should be added.
    - new_idr_amino_acids: List of 1-letter amino acid codes for the new IDR segment "ACDEF"
    - connecting_atom_name: The name of the atom in the first resolved residue to connect to (default 'CA').
    - start_ind: The starting residue index for the new IDR segment (default 1).
    - show_progress: Whether to display a progress bar during IDR construction (default True).
    - stiffness_angle: The bond angle in degrees for the random walk (default 120, where 180 is straight and 90 is a sharp turn).
    - bond_length: The distance between consecutive C-alpha atoms (default 3.8 Angstroms).
    - clash_distance: Minimum distance to avoid clashes with existing atoms (default 3.0 Angstroms).
    - attempts: Number of attempts to build the IDR if clashes are detected (default 5). If all attempts fail, an error is raised.
    - fake_build: If True, the function will simulate building without checking for clashes or generating realistic coordinates. This is useful for testing the integration of the build process without relying on the geometry functions.
    returns
    -------
    - modified_structure: A new Structure object with the N-terminal IDR added.
    """
    all_atoms = target_structure.get_coords()
    
    # Sort residues by ID to correctly identify the sequence N-terminus
    chain_residues = target_structure.chains[chain_id].residues
    def res_key(k):
        try: return int(k)
        except ValueError: return 999999
        
    sorted_keys = sorted(chain_residues.keys(), key=res_key)
    if not sorted_keys:
        raise ValueError(f"Chain {chain_id} has no residues.")
    
    first_ca_ind = sorted_keys[0]
    first_ca_coord = target_structure.chains[chain_id].residues[first_ca_ind][connecting_atom_name]
    first_ca_coordinates = np.array([first_ca_coord.x, first_ca_coord.y, first_ca_coord.z])
    for i in range(attempts):
        try:
            new_idr_atoms = build_idr_coordinates(
                connecting_atom_coords=first_ca_coordinates,
                num_residues=len(new_idr_amino_acids),
                current_coordinates=all_atoms,
                bond_length=bond_length,
                stiffness_angle=stiffness_angle,
                show_progress=show_progress,
                clash_distance=clash_distance,
                fake_build=fake_build
            )
            break
        except Exception as e:
            if i == attempts - 1:
                raise ValueError(
                    f"Failed to build N-terminal IDR for chain '{chain_id}' "
                    f"({len(new_idr_amino_acids)} residues) after {attempts} attempts. "
                    f"Last error: {e}. "
                    f"Try reducing stiffness_angle (current: {stiffness_angle}) or "
                    f"clash_distance (current: {clash_distance})."
                )
            else:
                logger.warning(f"Attempt {i+1}/{attempts} failed for N-terminal IDR on chain '{chain_id}': {e}. Retrying...")
                continue
    # reverse the order of the list from build_idr_coordinates since it builds outwards from the connecting atom, but for N-term we want to add in the opposite direction.
    new_idr_atoms = new_idr_atoms[::-1]
    res_names = [AA_MAP_1_TO_3[res] for res in new_idr_amino_acids]
    atom_names = ['CA'] * len(res_names)
    updated_struct = add_atoms_to_structure(target_structure, chain_id, new_idr_atoms, res_names, atom_names, start_ind=start_ind)
    return updated_struct


def build_loop(
        target_structure: Structure,
        chain_id: str,
        new_idr_amino_acids: str,
        ind_of_first_connecting_atom: int,
        ind_of_last_connecting_atom: int,
        connecting_atom_name: str = 'CA',
        show_progress: bool = True,
        stiffness_angle: float = 120,
        bond_length: float = 3.8,
        clash_distance: float = 3.0,
        attempts: int = 5,
        fake_build: bool = False
    ) -> Structure:
    """
    Docstring for build_n_term_idr
    
    parameters
    ----------
    - target_structure: Structure object representing the protein structure to modify.
    - chain_id: The ID of the chain to which the N-terminal IDR should be added.
    - new_idr_amino_acids: List of 1-letter amino acid codes for the new IDR segment "ACDEF"
    - connecting_atom_name: The name of the atom in the first resolved residue to connect to (default 'CA').
    - ind_of_first_connecting_atom: The index of the first connecting atom in the new IDR segment.
    - ind_of_last_connecting_atom: The index of the last connecting atom in the new IDR segment
    - show_progress: Whether to display a progress bar during IDR construction (default True).
    - stiffness_angle: The bond angle in degrees for the random walk (default 120, where 180 is straight and 90 is a sharp turn).
    - bond_length: The distance between consecutive C-alpha atoms (default 3.8 Angstroms).
    - clash_distance: Minimum distance to avoid clashes with existing atoms (default 3.0 Angstroms).     
    - attempts: Number of attempts to build the IDR if clashes are detected (default 5). If all attempts fail, an error is raised.   
    - fake_build: If True, the function will simulate building without checking for clashes or generating realistic coordinates. This is useful for testing the integration of the build process without relying on the geometry functions.
    returns
    -------
    - modified_structure: A new Structure object with the N-terminal IDR added.
    """
    all_atoms = target_structure.get_coords()
    
    # Sort residues by ID to correctly identify the sequence N-terminus
    chain_residues = target_structure.chains[chain_id].residues
    def res_key(k):
        try: return int(k)
        except ValueError: return 999999
        
    sorted_keys = sorted(chain_residues.keys(), key=res_key)
    if not sorted_keys:
        raise ValueError(f"Chain {chain_id} has no residues.")
    first_connecting_coord = target_structure.chains[chain_id].residues[str(ind_of_first_connecting_atom)][connecting_atom_name]
    first_connecting_coordinates = np.array([first_connecting_coord.x, first_connecting_coord.y, first_connecting_coord.z])
    last_connecting_coord = target_structure.chains[chain_id].residues[str(ind_of_last_connecting_atom)][connecting_atom_name]
    last_connecting_coordinates = np.array([last_connecting_coord.x, last_connecting_coord.y, last_connecting_coord.z]) 
    for i in range(attempts):
        try:
            new_idr_atoms = build_loop_coordinates(
                starting_coordinate=first_connecting_coordinates,
                ending_coordinate=last_connecting_coordinates,
                all_current_coordinates=all_atoms,
                num_residues=len(new_idr_amino_acids),
                start_index=int(ind_of_first_connecting_atom) + 1,
                bond_length=bond_length,
                clash_distance=clash_distance,
                show_progress=show_progress,
                fake_build=fake_build
            )
            break
        except Exception as e:
            if i == attempts - 1:
                raise ValueError(
                    f"Failed to build loop IDR for chain '{chain_id}' "
                    f"(residues {ind_of_first_connecting_atom+1}-{ind_of_last_connecting_atom-1}, "
                    f"{len(new_idr_amino_acids)} residues) after {attempts} attempts. "
                    f"Last error: {e}. "
                    f"Try reducing clash_distance (current: {clash_distance})."
                )
            else:
                logger.warning(f"Attempt {i+1}/{attempts} failed for loop IDR on chain '{chain_id}': {e}. Retrying...")
                continue
    res_names = [AA_MAP_1_TO_3[res] for res in new_idr_amino_acids]
    atom_names = ['CA'] * len(res_names)
    updated_struct = add_atoms_to_structure(target_structure, chain_id, new_idr_atoms, res_names, atom_names, start_ind=ind_of_first_connecting_atom + 1)
    return updated_struct


