"""
Docstring for protein_modifier.backend.build_idr
"""
from tqdm import tqdm
import numpy as np
from protein_modifier.backend.data_structures import Atom, Structure, Chain, Residue
from protein_modifier.backend.io import parse_cif, write_cif
from protein_modifier.backend.modify_structure import get_neighbors_in_sphere, get_centroid, generate_sphere_points, extend_line_segment, get_non_clashing_coords, generate_next_calpha
from protein_modifier.data.amino_acids import AA_MAP_1_TO_3
from protein_modifier.backend.protein_math import calculate_distance, find_furthest_coordinate, find_points_within_sphere

def build_idr_coordinates(
        connecting_atom_coords: np.ndarray,
        num_residues: int,
        current_coordinates: np.ndarray,
        bond_length: float = 3.8,
        stiffness_angle: float = 120,
        show_progress: bool = True,
        clash_distance: float = 3.0
    ):
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
    Returns:
    - List of new atom dicts with keys: x, y, z
    """
    # list to hold new atoms.
    new_atoms = []
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
    # set initial radius to distance between start and end + bond_length
    radius = np.linalg.norm(ending_coordinate - starting_coordinate) + bond_length
    # precompute all radii based on number of residues
    radii = np.linspace(radius, bond_length, num_residues)
    # manually determined values that work really well for the last 6 residues. 
    manual_dist=[15.8868, 13.6407, 11.0914, 8.6688, 6.441, 3.856]
    # replace final five values of radii with these.
    if len(manual_dist) <= num_residues:
        radii[-len(manual_dist):] = manual_dist
    else:
        manual_dist = manual_dist[-num_residues:]
        radii = manual_dist
    current_coords = all_current_coordinates.copy()
    
    new_coords = []
    for i in tqdm(range(num_residues), disable=not show_progress):
        radius = radii[i]
        # generate sphere points from start coordinate at bond length distance
        sphere_points = generate_sphere_points(starting_coordinate, radius=bond_length, num_points=5000)
        # filter out points that clash with current structure
        candidates = get_non_clashing_coords(sphere_points, current_coords, min_distance=clash_distance)
        if len(candidates) == 0:
            raise ValueError(f"Could not find non-clashing candidates for loop residue {start_index + i}")
        # get points within sphere with radius=radius
        candidates = find_points_within_sphere(candidates, ending_coordinate, radius)
        # calculate distances to ending coordinate
        dists = np.linalg.norm(candidates - ending_coordinate, axis=1)
        # select distance closest to radius value
        diff_to_cur_radius = np.abs(dists - radius)
        best_index = np.argmin(diff_to_cur_radius)
        final_coord = candidates[best_index]
        new_coords.append(final_coord)
        current_coords = np.vstack((current_coords, final_coord))
        # reduce radius for next iteration to encourage moving towards the end coordinate
        radius = radius - bond_length
        starting_coordinate = final_coord
    return np.array(new_coords)

def add_atoms_to_structure(structure, chain_id, new_atoms, residue_names, atom_names, start_ind=1):
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
        residue.add_atom(Atom(atom_dict))
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
        clash_distance: float = 3.0
    ):
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
    new_idr_atoms = build_idr_coordinates(
        connecting_atom_coords=last_ca_coordinates,
        num_residues=len(new_idr_amino_acids),
        current_coordinates=all_atoms,
        bond_length=bond_length,
        stiffness_angle=stiffness_angle,
        show_progress=show_progress,
        clash_distance=clash_distance
    )
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
        clash_distance: float = 3.0
    ):
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
    new_idr_atoms = build_idr_coordinates(
        connecting_atom_coords=first_ca_coordinates,
        num_residues=len(new_idr_amino_acids),
        current_coordinates=all_atoms,
        bond_length=bond_length,
        stiffness_angle=stiffness_angle,
        show_progress=show_progress,
        clash_distance=clash_distance
    )
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
        clash_distance: float = 3.0
    ):
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
    new_idr_atoms = build_loop_coordinates(
        starting_coordinate=first_connecting_coordinates,
        ending_coordinate=last_connecting_coordinates,
        all_current_coordinates=all_atoms,
        num_residues=len(new_idr_amino_acids),
        start_index=int(ind_of_first_connecting_atom) + 1,
        bond_length=bond_length,
        clash_distance=clash_distance,
        show_progress=show_progress
    )
    res_names = [AA_MAP_1_TO_3[res] for res in new_idr_amino_acids]
    atom_names = ['CA'] * len(res_names)
    updated_struct = add_atoms_to_structure(target_structure, chain_id, new_idr_atoms, res_names, atom_names, start_ind=ind_of_first_connecting_atom + 1)
    return updated_struct

"""
# usage examples...
# now to test it out. 
test_cif = parse_cif("/Users/ryanemenecker/Desktop/lab_packages/protein_modifier/protein_modifier/data/6KN8-assembly1-coarse-grained.cif")
updated_struct = Structure.from_dict(test_cif)


n_amino_acids = 'MSDIEEVVEEYEEEEQEEAAVEEQEEAAEEDAEAEAETEETRAEEDEEEEEAKEAEDGPMEESKPKPRSFMPNLVPPKIPDGERVDFDDIHRKRMEKD'
updated_struct = build_n_term_idr(
    target_structure=updated_struct,
    chain_id='T',
    new_idr_amino_acids=n_amino_acids,
    connecting_atom_name='CA',
    start_ind=1,
    show_progress=True,
    stiffness_angle=150,
    bond_length=3.8,
    clash_distance=3.4
)


c_amino_acids = 'KVSKTRGKAKVTGRWK'
updated_struct = build_c_term_idr(
    target_structure=updated_struct,
    chain_id='T',
    new_idr_amino_acids=c_amino_acids,
    connecting_atom_name='CA',
    start_ind=None,
    show_progress=True,
    stiffness_angle=150,
    bond_length=3.8,
    clash_distance=3.4
)

loop_amino_acids = "RLAEERARREEEENRRKAEDEARKKKALSNMMHFGGYIQKQAQTERKS"
updated_struct = build_loop(
    target_structure=updated_struct,
    chain_id='T',
    new_idr_amino_acids=loop_amino_acids,
    ind_of_first_connecting_atom=150,
    ind_of_last_connecting_atom=199,
    connecting_atom_name='CA',
    show_progress=True,
    stiffness_angle=150,
    bond_length=3.8,
    clash_distance=3.4
)


write_cif(updated_struct.to_dict(), "/Users/ryanemenecker/Desktop/lab_packages/protein_modifier/protein_modifier/data/6KN8-assembly1-with-idr.cif")
"""


