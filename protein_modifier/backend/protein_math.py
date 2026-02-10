'''
holds common math and math-related functions.
'''
import numpy as np

def calculate_distance(coord1, coord2):
    '''
    This function calculates the Euclidean distance between two points.
    
    Parameters
    ----------
    coord1 : np.array, shape=(3,)
        coordinates of the first point
    coord2 : np.array, shape=(3,)
        coordinates of the second point
    
    Returns
    -------
    float
        Euclidean distance between the two points
    '''
    return np.linalg.norm(coord1 - coord2)

def calculate_distance_vectorized(coord_list1, coord_list2):
    '''
    This function calculates the Euclidean distance between two lists of points.
    
    Parameters
    ----------
    coord_list1 : np.array, shape=(N, 3)
        coordinates of the first list of points
    coord_list2 : np.array, shape=(M, 3)
        coordinates of the second list of points
    
    Returns
    -------
    np.array, shape=(N, M)
        Euclidean distance between each pair of points
    '''
    return np.linalg.norm(coord_list1[:, np.newaxis, :] - coord_list2[np.newaxis, :, :], axis=2)



def find_furthest_coordinate(coord_list1, coord_list2):
    '''
    Finds the coordinate in the first list that is furthest from all coordinates in the second list
    using vectorized operations for efficiency.
    
    Parameters
    ----------
    coord_list1 : array-like, shape=(N, 3)
        First list of coordinates to check
    coord_list2 : array-like, shape=(M, 3)
        Second list of coordinates to compare against
    
    Returns
    -------
    furthest_coord : numpy.ndarray, shape=(3,)
        The coordinate from coord_list1 that has the maximum minimum distance to any point in coord_list2
    max_min_distance : float
        The minimum distance from the returned coordinate to any point in coord_list2
    
    Raises
    ------
    ValueError
        If inputs have incorrect shapes or are empty
    '''
    try:
        points1 = np.asarray(coord_list1, dtype=float)
        points2 = np.asarray(coord_list2, dtype=float)
    except (ValueError, TypeError):
        raise ValueError("Coordinates must be convertible to float arrays")

    # Validate input shapes
    if points1.ndim != 2 or points1.shape[1] != 3:
        raise ValueError("coord_list1 must be an Nx3 array")
    if points2.ndim != 2 or points2.shape[1] != 3:
        raise ValueError("coord_list2 must be an Mx3 array")
    if len(points1) == 0 or len(points2) == 0:
        raise ValueError("Coordinate lists cannot be empty")

    # Vectorized distance calculation
    # Reshape arrays for broadcasting: (N, 1, 3) - (1, M, 3) -> (N, M, 3)
    diff = points1[:, np.newaxis, :] - points2[np.newaxis, :, :]
    
    # Calculate distances efficiently using einsum
    # equivalent to np.sqrt(np.sum(diff * diff, axis=2))
    distances = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
    
    # Find minimum distance for each point in points1
    min_distances = np.min(distances, axis=1)
    
    # Find the point with the maximum minimum distance
    max_min_idx = np.argmax(min_distances)
    
    return points1[max_min_idx]



def find_points_within_sphere(points_of_interest, sphere_center, sphere_radius):
    '''
    Find all points that lie within a sphere of given radius and center.
    
    Parameters
    ----------
    points_of_interest : array-like, shape=(N, 3)
        Array of points to check. Each point should be a 3D coordinate.
    sphere_center : array-like, shape=(3,)
        Center coordinates of the sphere
    sphere_radius : float
        Radius of the sphere. Must be positive.
    
    Returns
    -------
    numpy.ndarray
        Array containing only the points that lie within the sphere
    points_mask : numpy.ndarray
        Boolean mask indicating which points are within the sphere
    
    Raises
    ------
    ValueError
        If inputs have incorrect shapes or if radius is not positive
    '''
    # Convert inputs to numpy arrays and validate
    try:
        points = np.asarray(points_of_interest, dtype=float)
        center = np.asarray(sphere_center, dtype=float)
        radius = float(sphere_radius)
    except (ValueError, TypeError):
        raise ValueError("All inputs must be convertible to float types")

    # Check shapes
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_of_interest must be an Nx3 array")
    if center.shape != (3,):
        raise ValueError("sphere_center must be a 3D point")
    if radius <= 0:
        raise ValueError("sphere_radius must be positive")

    # Vectorized distance calculation
    # Using einsum for better memory efficiency with large arrays
    # equivalent to: distances = np.sqrt(np.sum((points - center)**2, axis=1))
    distances = np.sqrt(np.einsum('ij,ij->i', points - center, points - center))
    
    # Return both the points and the mask
    return points[distances <= radius]



def find_points_not_clashing(potential_coords, coords_to_check, clash_distance=3):
    '''
    This function finds the points in potential_coords that are not clashing with any points in coords_to_check.
    
    Parameters
    ----------
    potential_coords : np.array, shape=(N, P, 3)
        Points to check for clashes. N is number of sets, P is points per set.
    coords_to_check : np.array, shape=(M, 3)
        Points to check against.
    clash_distance : float
        Distance at which two points are considered clashing.
    
    Returns
    -------
    np.array, shape=(K, 3)
        Points that are not clashing.
    '''
    # Reshape arrays for broadcasting
    potential_coords = np.asarray(potential_coords)
    coords_to_check = np.asarray(coords_to_check)
    
    # Calculate distances between all points
    # Reshape arrays to allow broadcasting
    distances = np.linalg.norm(
        potential_coords[..., np.newaxis, :] - coords_to_check, 
        axis=-1
    )
    
    # Find points that don't clash with any point in coords_to_check
    non_clashing_mask = np.all(distances > clash_distance, axis=-1)
    non_clashing_points = potential_coords[non_clashing_mask]
    
    return non_clashing_points