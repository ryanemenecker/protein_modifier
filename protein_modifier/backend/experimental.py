"""Experimental path generators for difficult loop-closure cases."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _as_coord_array(existing_coords: list[tuple[float, float, float]] | np.ndarray) -> np.ndarray:
    if existing_coords is None or len(existing_coords) == 0:
        return np.empty((0, 3), dtype=float)
    coords = np.asarray(existing_coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("existing_coords must be convertible to an Nx3 array")
    return coords


def _build_fibonacci_sphere(num_points: int) -> np.ndarray:
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_points)
    theta = np.pi * (1 + 5**0.5) * indices
    return np.column_stack((
        np.cos(theta) * np.sin(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(phi),
    ))


def _sphere_intersection_points(
        center_a: np.ndarray,
        center_b: np.ndarray,
        radius_a: float,
        radius_b: float,
        num_points: int) -> np.ndarray:
    center_a = np.asarray(center_a, dtype=float)
    center_b = np.asarray(center_b, dtype=float)
    distance = np.linalg.norm(center_b - center_a)
    tolerance = 1e-6

    if distance < tolerance:
        return np.empty((0, 3), dtype=float)
    if distance > (radius_a + radius_b + tolerance):
        return np.empty((0, 3), dtype=float)
    if distance < (abs(radius_a - radius_b) - tolerance):
        return np.empty((0, 3), dtype=float)

    circle_center_distance = (radius_a ** 2 - radius_b ** 2 + distance ** 2) / (2 * distance)
    circle_radius_sq = radius_a ** 2 - circle_center_distance ** 2
    if circle_radius_sq < 0:
        if circle_radius_sq > -1e-6:
            circle_radius_sq = 0.0
        else:
            return np.empty((0, 3), dtype=float)

    axis = (center_b - center_a) / distance
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


def _filter_non_clashing(candidate_points: np.ndarray, occupied_coords: np.ndarray, clash_dist: float) -> np.ndarray:
    if len(candidate_points) == 0:
        return candidate_points
    if len(occupied_coords) == 0:
        return candidate_points

    tree = cKDTree(occupied_coords)
    distances, _ = tree.query(candidate_points, k=1)
    return candidate_points[distances >= clash_dist]


def _candidate_clearance(candidate_points: np.ndarray, occupied_coords: np.ndarray) -> np.ndarray:
    if len(candidate_points) == 0:
        return np.array([])
    if len(occupied_coords) == 0:
        return np.full(len(candidate_points), np.inf)

    tree = cKDTree(occupied_coords)
    distances, _ = tree.query(candidate_points, k=1)
    return distances


def _generate_step_candidates(
        current_coord: np.ndarray,
        end_coord: np.ndarray,
        occupied_coords: np.ndarray,
        steps_remaining: int,
        target_dist: float,
        clash_dist: float,
        unit_sphere: np.ndarray) -> np.ndarray:
    if steps_remaining == 1:
        candidates = _sphere_intersection_points(
            current_coord,
            end_coord,
            target_dist,
            target_dist,
            len(unit_sphere),
        )
        return _filter_non_clashing(candidates, occupied_coords, clash_dist)

    if steps_remaining == 2:
        candidates = _sphere_intersection_points(
            current_coord,
            end_coord,
            target_dist,
            2 * target_dist,
            len(unit_sphere),
        )
        return _filter_non_clashing(candidates, occupied_coords, clash_dist)

    candidates = current_coord + (unit_sphere * target_dist)
    candidates = _filter_non_clashing(candidates, occupied_coords, clash_dist)
    if len(candidates) == 0:
        return candidates

    max_reachable_distance = steps_remaining * target_dist + 1e-6
    end_distances = np.linalg.norm(candidates - end_coord, axis=1)
    reachable = candidates[end_distances <= max_reachable_distance]
    if len(reachable) > 0:
        return reachable

    # If the reachable cap is very narrow, exact sphere intersections can recover points
    intersection_points = _sphere_intersection_points(
        current_coord,
        end_coord,
        target_dist,
        max_reachable_distance,
        len(unit_sphere),
    )
    return _filter_non_clashing(intersection_points, occupied_coords, clash_dist)


def _score_candidates(
        candidate_points: np.ndarray,
        end_coord: np.ndarray,
        occupied_coords: np.ndarray,
        desired_end_distance: float) -> np.ndarray:
    end_distances = np.linalg.norm(candidate_points - end_coord, axis=1)
    clearances = _candidate_clearance(candidate_points, occupied_coords)
    return (0.20 * clearances) - np.abs(end_distances - desired_end_distance)


def _search_small_gap(
        start_coord: np.ndarray,
        end_coord: np.ndarray,
        occupied_coords: np.ndarray,
        num_bridge_coords: int,
        target_dist: float,
        clash_dist: float,
        unit_sphere: np.ndarray,
        branch_factor: int) -> np.ndarray | None:
    if num_bridge_coords == 0:
        if abs(np.linalg.norm(end_coord - start_coord) - target_dist) <= 1e-3:
            return np.empty((0, 3))
        return None

    candidates = _generate_step_candidates(
        start_coord,
        end_coord,
        occupied_coords,
        num_bridge_coords,
        target_dist,
        clash_dist,
        unit_sphere,
    )
    if len(candidates) == 0:
        return None

    desired_end_distance = num_bridge_coords * target_dist
    candidate_scores = _score_candidates(candidates, end_coord, occupied_coords, desired_end_distance)
    top_indices = np.argsort(candidate_scores)[-branch_factor:][::-1]

    for idx in top_indices:
        candidate = candidates[idx]
        next_occupied = np.vstack((occupied_coords, candidate))
        remainder = _search_small_gap(
            candidate,
            end_coord,
            next_occupied,
            num_bridge_coords - 1,
            target_dist,
            clash_dist,
            unit_sphere,
            branch_factor,
        )
        if remainder is not None:
            return np.vstack((candidate.reshape(1, 3), remainder)) if len(remainder) else candidate.reshape(1, 3)

    return None


def _build_frontier(
        start_coord: np.ndarray,
        end_coord: np.ndarray,
        occupied_coords: np.ndarray,
        num_steps: int,
        remaining_bridge_coords: int,
        target_dist: float,
        clash_dist: float,
        unit_sphere: np.ndarray,
        branch_factor: int,
        beam_width: int) -> list[tuple[list[np.ndarray], np.ndarray, np.ndarray]]:
    frontier: list[tuple[list[np.ndarray], np.ndarray, np.ndarray]] = [([], start_coord, occupied_coords)]

    for step_index in range(num_steps):
        steps_remaining = (num_steps - step_index) + remaining_bridge_coords
        next_states = []

        for path, current_coord, current_occupied in frontier:
            candidates = _generate_step_candidates(
                current_coord,
                end_coord,
                current_occupied,
                steps_remaining,
                target_dist,
                clash_dist,
                unit_sphere,
            )
            if len(candidates) == 0:
                continue

            desired_end_distance = steps_remaining * target_dist
            candidate_scores = _score_candidates(candidates, end_coord, current_occupied, desired_end_distance)
            top_indices = np.argsort(candidate_scores)[-branch_factor:][::-1]

            for idx in top_indices:
                candidate = candidates[idx]
                next_path = path + [candidate]
                next_occupied = np.vstack((current_occupied, candidate))
                next_states.append((candidate_scores[idx], next_path, candidate, next_occupied))

        if not next_states:
            return []

        next_states.sort(key=lambda item: item[0], reverse=True)
        frontier = [
            (path, current_coord, current_occupied)
            for _, path, current_coord, current_occupied in next_states[:beam_width]
        ]

    return frontier


def _iter_bidirectional_splits(num_coords: int):
    seen = set()
    max_middle_coords = min(4, num_coords - 2)

    for middle_coords in range(0, max_middle_coords + 1):
        remaining_coords = num_coords - middle_coords
        base_left_steps = remaining_coords // 2

        for delta in (0, -1, 1, -2, 2):
            left_steps = base_left_steps + delta
            right_steps = remaining_coords - left_steps
            if left_steps < 1 or right_steps < 1:
                continue

            split = (left_steps, middle_coords, right_steps)
            if split in seen:
                continue

            seen.add(split)
            yield split


def _generate_bidirectional_path(
        start_coord: np.ndarray,
        end_coord: np.ndarray,
        occupied_coords: np.ndarray,
        num_coords: int,
        target_dist: float,
        clash_dist: float,
        unit_sphere: np.ndarray,
        branch_factor: int,
        beam_width: int) -> np.ndarray | None:
    if num_coords < 2:
        return None

    for left_steps, middle_coords, right_steps in _iter_bidirectional_splits(num_coords):
        left_frontier = _build_frontier(
            start_coord,
            end_coord,
            occupied_coords,
            left_steps,
            right_steps + middle_coords,
            target_dist,
            clash_dist,
            unit_sphere,
            branch_factor,
            beam_width,
        )
        if not left_frontier:
            continue

        right_frontier = _build_frontier(
            end_coord,
            start_coord,
            occupied_coords,
            right_steps,
            left_steps + middle_coords,
            target_dist,
            clash_dist,
            unit_sphere,
            branch_factor,
            beam_width,
        )
        if not right_frontier:
            continue

        for left_path, left_coord, _ in left_frontier:
            left_array = np.array(left_path) if left_path else np.empty((0, 3))
            for right_path, right_coord, _ in right_frontier:
                right_forward = right_path[::-1]
                right_array = np.array(right_forward) if right_forward else np.empty((0, 3))

                pair_occupied = occupied_coords
                if len(left_array):
                    pair_occupied = np.vstack((pair_occupied, left_array))
                if len(right_array):
                    pair_occupied = np.vstack((pair_occupied, right_array))

                bridge = _search_small_gap(
                    left_coord,
                    right_coord,
                    pair_occupied,
                    middle_coords,
                    target_dist,
                    clash_dist,
                    unit_sphere,
                    branch_factor,
                )
                if bridge is None:
                    continue

                pieces = []
                if len(left_array):
                    pieces.append(left_array)
                if len(bridge):
                    pieces.append(bridge)
                if len(right_array):
                    pieces.append(right_array)
                if not pieces:
                    return np.empty((0, 3))
                return np.vstack(pieces)

    return None


def generate_directed_path(
    start_coord: tuple[float, float, float],
    end_coord: tuple[float, float, float],
    existing_coords: list[tuple[float, float, float]] | np.ndarray,
    num_coords: int,
    target_dist: float,
    clash_dist: float,
    num_sphere_points: int = 4000,
    branch_factor: int = 20,
    beam_width: int = 160,
) -> np.ndarray:
    """Generate an intermediate path using reachability-pruned beam search.

    The search keeps multiple non-clashing partial paths alive, prunes candidates
    that cannot possibly reach the end anchor in the remaining number of fixed-
    length steps, and uses exact sphere intersections for the last two steps.
    """
    if num_coords < 1:
        raise ValueError("num_coords must be at least 1")
    if target_dist <= 0:
        raise ValueError("target_dist must be positive")
    if clash_dist <= 0:
        raise ValueError("clash_dist must be positive")
    if num_sphere_points < 32:
        raise ValueError("num_sphere_points must be at least 32")

    start_coord = np.asarray(start_coord, dtype=float)
    end_coord = np.asarray(end_coord, dtype=float)
    occupied_coords = _as_coord_array(existing_coords)
    if len(occupied_coords) == 0:
        occupied_coords = start_coord.reshape(1, 3)
    else:
        occupied_coords = np.vstack((occupied_coords, start_coord))

    anchor_distance = np.linalg.norm(end_coord - start_coord)
    max_bridge_distance = (num_coords + 1) * target_dist
    if anchor_distance > max_bridge_distance + 1e-6:
        raise RuntimeError(
            f"Anchors are {anchor_distance:.4f} A apart, but {num_coords} intermediate points "
            f"with spacing {target_dist:.4f} A can span at most {max_bridge_distance:.4f} A."
        )

    unit_sphere = _build_fibonacci_sphere(num_sphere_points)
    beam: list[tuple[list[np.ndarray], np.ndarray, np.ndarray]] = [([], start_coord, occupied_coords)]
    beam_error: RuntimeError | None = None

    for step_index in range(num_coords):
        steps_remaining = num_coords - step_index
        next_states = []

        for path, current_coord, current_occupied in beam:
            candidates = _generate_step_candidates(
                current_coord,
                end_coord,
                current_occupied,
                steps_remaining,
                target_dist,
                clash_dist,
                unit_sphere,
            )
            if len(candidates) == 0:
                continue

            desired_end_distance = steps_remaining * target_dist
            candidate_scores = _score_candidates(candidates, end_coord, current_occupied, desired_end_distance)
            top_indices = np.argsort(candidate_scores)[-branch_factor:][::-1]

            for idx in top_indices:
                candidate = candidates[idx]
                new_path = path + [candidate]
                new_occupied = np.vstack((current_occupied, candidate))
                penalty = 0.05 * np.linalg.norm(candidate - end_coord) * (steps_remaining - 1)
                next_states.append((candidate_scores[idx] - penalty, new_path, candidate, new_occupied))

        if not next_states:
            beam_error = RuntimeError(
                f"No reachable non-clashing candidates remained at step {step_index + 1}. "
                "Consider decreasing clash_dist or increasing beam_width/num_sphere_points."
            )
            break

        next_states.sort(key=lambda item: item[0], reverse=True)
        beam = [
            (path, current_coord, current_occupied)
            for _, path, current_coord, current_occupied in next_states[:beam_width]
        ]

    if beam_error is None:
        beam_result = np.array(beam[0][0])
        final_distance = np.linalg.norm(beam_result[-1] - end_coord)
        if abs(final_distance - target_dist) <= 1e-3:
            return beam_result

    bidirectional_result = _generate_bidirectional_path(
        start_coord,
        end_coord,
        occupied_coords,
        num_coords,
        target_dist,
        clash_dist,
        unit_sphere,
        branch_factor,
        beam_width,
    )
    if bidirectional_result is not None:
        return bidirectional_result

    if beam_error is not None:
        raise beam_error
    return beam_result

