"""
Docstring for protein_modifier.backend.modify_structure
"""

import sys
import os
import numpy as np
from scipy.spatial.distance import cdist

# code for random(ish) walk
# tries to capture the fact that backbones have a stiffness constraint. 

def normalize(v):
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def rotation_matrix_from_vectors(vec1, vec2):
    """
    Returns the rotation matrix that rotates vec1 to align with vec2.
    Using the Rodrigues' rotation formula.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    
    # Check for parallel vectors to avoid division by zero
    if s == 0:
        # If parallel, return identity (or flip if anti-parallel, not handled here for simplicity
        # as it's rare in this specific random walk context)
        return np.eye(3)

    kmat = np.array([[0, -v[2], v[1]], 
                     [v[2], 0, -v[0]], 
                     [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def generate_next_calpha(prev_pos, prev_prev_pos, bond_length=3.8, stiffness_angle=120):
    """
    Generates the next C-alpha position.
    
    Parameters:
    - prev_pos: The (x,y,z) of the current end of the chain.
    - prev_prev_pos: The (x,y,z) of the atom before the end (needed for direction).
    - bond_length: Distance to next atom (default 3.8 Angstroms for CA-CA).
    - stiffness_angle: The bond angle in degrees (angle between p_prev-p_pprev and new_vec).
                       180 = perfectly straight chain.
                       90 = sharp turn.
    """
    prev_pos = np.array(prev_pos)
    prev_prev_pos = np.array(prev_prev_pos)
    
    # 1. Define the direction of the previous bond (this becomes our local Z-axis)
    prev_bond_vec = normalize(prev_pos - prev_prev_pos)
    
    # 2. Convert constraints to radians
    # Note: In spherical coords aligned to Z, theta is angle from Z-axis.
    # If stiffness_angle is 180 (straight), theta deviation is 0.
    # If stiffness_angle is 120 (bent), theta deviation is 60 (180-120).
    theta_rad = np.radians(180 - stiffness_angle) 
    
    # 3. Generate Random Torsion (Phi) - The "Random" part of the walk
    phi_rad = np.random.uniform(0, 2 * np.pi)
    
    # 4. Calculate the new vector in a Local Coordinate System
    # Assume the previous bond is aligned with the Z-axis [0,0,1]
    # x = r * sin(theta) * cos(phi)
    # y = r * sin(theta) * sin(phi)
    # z = r * cos(theta)
    
    local_x = bond_length * np.sin(theta_rad) * np.cos(phi_rad)
    local_y = bond_length * np.sin(theta_rad) * np.sin(phi_rad)
    local_z = bond_length * np.cos(theta_rad)
    
    local_next_vec = np.array([local_x, local_y, local_z])
    
    # 5. Rotate this local vector to align with the actual previous bond
    # We want to rotate local Z-axis [0,0,1] to match prev_bond_vec
    z_axis = np.array([0, 0, 1])
    rot_mat = rotation_matrix_from_vectors(z_axis, prev_bond_vec)
    
    real_next_vec = np.dot(rot_mat, local_next_vec)
    
    # 6. Add to previous position
    new_pos = prev_pos + real_next_vec
    
    return new_pos

def get_non_clashing_coords(candidates, obstacles, min_distance):
    """
    Returns a subset of 'candidates' that are at least 'min_distance'
    away from ALL points in 'obstacles'.
    
    Parameters:
    - candidates: (N, 3) np.array of new points you want to test.
    - obstacles: (M, 3) np.array of existing points (e.g., the rest of the protein).
    - min_distance: float, the safety radius (e.g., 3.0 Angstroms).
    """
    
    # Ensure candidates is a 2D array (N, 3)
    candidates = np.array(candidates)
    if candidates.ndim == 1:
        candidates = candidates.reshape(1, -1)
    
    # 1. Calculate the Euclidean distance between every candidate and every obstacle.
    # Result is a (N x M) matrix where [i, j] is dist between candidate[i] and obstacle[j].
    # 'euclidean' is the default metric, but being explicit is good practice.
    dists = cdist(candidates, obstacles, metric='euclidean')
    
    # 2. For each candidate (row), find the distance to its *nearest* obstacle.
    # axis=1 operates across columns (checking all obstacles for one candidate)
    nearest_obstacle_dist = dists.min(axis=1)
    
    # 3. Create a boolean mask: True if the nearest obstacle is far enough away.
    valid_mask = nearest_obstacle_dist >= min_distance
    
    # 4. Filter the original array
    clean_candidates = candidates[valid_mask]
    
    return clean_candidates

# needed to get away from the filament for initial steps.
def extend_line_segment(point_a, point_b, distance):
    """
    Generates a point C that lies on the line passing through A and B.
    C is placed 'distance' units away from B, in the direction of A->B.
    
    Parameters:
    - point_a: Starting point of the vector (numpy array or list)
    - point_b: Ending point of the vector (numpy array or list)
    - distance: How far from B to place C. 
                Positive values extend OUTWARDS (away from A).
                Negative values extend BACKWARDS (towards A).
    """
    a = np.array(point_a)
    b = np.array(point_b)
    
    # 1. Calculate the vector from A to B
    vector_ab = b - a
    
    # 2. Calculate the length (norm) of that vector
    norm = np.linalg.norm(vector_ab)
    
    # Safety check: avoid division by zero if A and B are the same point
    if norm == 0:
        raise ValueError("Point A and Point B are identical; direction is undefined.")
    
    # 3. Calculate the unit vector (direction)
    unit_vector = vector_ab / norm
    
    # 4. Calculate Point C
    # Start at B, move along the unit vector by 'distance'
    point_c = b + (unit_vector * distance)
    
    return point_c

def generate_sphere_points(center, radius, num_points):
    """
    Generates 'num_points' randomly distributed on the surface of a sphere.
    
    Parameters:
    - center: (3,) array or list [x, y, z]
    - radius: float
    - num_points: int
    
    Returns:
    - points: (num_points, 3) numpy array
    """
    center = np.array(center)
    
    # 1. Generate random variables
    # phi: Azimuthal angle (0 to 2pi) -> uniform distribution
    # costheta: Cosine of polar angle (-1 to 1) -> uniform distribution to avoid clumping at poles
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    costheta = np.random.uniform(-1, 1, num_points)
    
    # 2. Calculate derived trigonometric values
    # theta = arccos(costheta), so sin(theta) = sqrt(1 - costheta^2)
    theta = np.arccos(costheta)
    sintheta = np.sin(theta)
    
    # 3. Convert Spherical to Cartesian coordinates
    # x = r * sin(theta) * cos(phi)
    # y = r * sin(theta) * sin(phi)
    # z = r * cos(theta)
    
    x = radius * sintheta * np.cos(phi)
    y = radius * sintheta * np.sin(phi)
    z = radius * costheta
    
    # 4. Stack and translate to the specified center
    # axis=-1 stacks them as columns (N, 3)
    points = np.stack((x, y, z), axis=-1)
    
    return points + center


def get_neighbors_in_sphere(center, coords_list, radius):
    """
    Identifies coordinates within a specified radius of a center point.
    
    Args:
        center (tuple/list): The (x, y, z) of the sphere's origin.
        coords_list (list): A list of (x, y, z) tuples to check.
        radius (float): The radius of the sphere.
        
    Returns:
        list: A subset of coords_list containing only points inside the sphere.
    """
    cx, cy, cz = center
    radius_sq = radius * radius
    
    neighbors = []
    
    for point in coords_list:
        px, py, pz = point
        
        # Calculate squared Euclidean distance
        # (x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2
        dist_sq = (px - cx)**2 + (py - cy)**2 + (pz - cz)**2
        
        if dist_sq <= radius_sq:
            neighbors.append(point)
            
    return neighbors

def get_centroid(coords_list):
    """
    Calculates the geometric center (centroid) of a list of coordinates.
    
    This point is the 'center of mass' of the coordinates. It is the 
    best approximation for a point equidistant to the entire set.
    
    Args:
        coords_list (list): A list of (x, y, z) tuples.
        
    Returns:
        tuple: (x, y, z) of the centroid. Returns None if list is empty.
    """
    if not coords_list:
        return None
        
    # 1. Count the points
    n = len(coords_list)
    
    # 2. Sum the dimensions
    sum_x = sum(p[0] for p in coords_list)
    sum_y = sum(p[1] for p in coords_list)
    sum_z = sum(p[2] for p in coords_list)
    
    # 3. Divide by N to get the average
    return (sum_x / n, sum_y / n, sum_z / n)