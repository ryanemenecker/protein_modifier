"""
Unit and regression test for the protein_modifier package.
"""

import sys
import os
import json
import tempfile
from unittest import mock
import numpy as np
import pytest

import protein_modifier
import protein_modifier.cli as cli
import protein_modifier.backend.sim_file_generation as sim_file_generation
from protein_modifier.backend import build_idr
from protein_modifier.backend import experimental
from protein_modifier.backend.combine import combine_structures
from protein_modifier.backend.io import parse_cif, write_cif, parse_pdb, write_pdb, parse_structure
from protein_modifier.backend.data_structures import Atom, Residue, Chain, Structure
from protein_modifier.data.elements import ELEMENT_MASSES
from protein_modifier.backend.find_missing_res import (
    affine_global_align,
    get_missing_residues_by_number,
)
from protein_modifier.backend.parse_build_file import read_build_file, set_up_data
from protein_modifier.backend.sim_file_generation import get_lammps_group_numbers
from protein_modifier.backend.modify_structure import (
    generate_next_calpha,
    rotation_matrix_from_vectors,
    get_non_clashing_coords,
    generate_sphere_points,
    get_neighbors_in_sphere,
    get_centroid,
    kabsch_align,
)
from protein_modifier.backend.protein_math import (
    calculate_distance,
    find_points_within_sphere,
)

# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CG_CIF = os.path.join(DATA_DIR, "6KN8-assembly1-coarse-grained.cif")
ALLATOM_CIF = os.path.join(DATA_DIR, "6KN8-assembly1.cif")


@pytest.fixture
def cg_structure():
    """Load the coarse-grained test CIF as a Structure."""
    return Structure.from_dict(parse_cif(CG_CIF))


@pytest.fixture
def allatom_structure():
    """Load the all-atom test CIF as a Structure."""
    return Structure.from_dict(parse_cif(ALLATOM_CIF))


# ──────────────────────────────────────────────
# 0. Import smoke test
# ──────────────────────────────────────────────

def test_protein_modifier_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "protein_modifier" in sys.modules


# ──────────────────────────────────────────────
# 1. CIF Parsing
# ──────────────────────────────────────────────

class TestCIFParsing:
    def test_parse_cif_returns_dict(self):
        result = parse_cif(CG_CIF)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_parse_cif_chains_present(self):
        result = parse_cif(CG_CIF)
        assert "A" in result

    def test_parse_cif_residues_have_atoms(self):
        result = parse_cif(CG_CIF)
        chain_a = result["A"]
        first_res_key = list(chain_a.keys())[0]
        atoms = chain_a[first_res_key]
        assert len(atoms) > 0
        assert "Cartn_x" in atoms[0]

    def test_parse_cif_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_cif("/nonexistent/path.cif")

    def test_parse_structure_auto_detect_cif(self):
        result = parse_structure(CG_CIF)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_parse_structure_unsupported_format(self):
        with pytest.raises(ValueError, match="Unsupported file format"):
            parse_structure("test.xyz")


# ──────────────────────────────────────────────
# 2. CIF/PDB Round-Trip
# ──────────────────────────────────────────────

class TestRoundTrip:
    def test_cif_write_read_roundtrip(self, cg_structure):
        """Parse CIF → Structure → to_dict → write CIF → re-parse → compare."""
        with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as f:
            tmp_path = f.name
        try:
            write_cif(cg_structure.to_dict(), tmp_path)
            reloaded = Structure.from_dict(parse_cif(tmp_path))
            # Same number of chains
            assert set(cg_structure.chains.keys()) == set(reloaded.chains.keys())
            # Same number of residues per chain
            for chain_id in cg_structure.chains:
                orig_count = len(cg_structure.chains[chain_id].residues)
                new_count = len(reloaded.chains[chain_id].residues)
                assert orig_count == new_count, f"Chain {chain_id}: {orig_count} != {new_count}"
        finally:
            os.unlink(tmp_path)

    def test_pdb_write_read_roundtrip(self, cg_structure):
        """Parse CIF → Structure → to_dict → write PDB → parse PDB → compare."""
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            tmp_path = f.name
        try:
            write_pdb(cg_structure.to_dict(), tmp_path)
            reloaded = Structure.from_dict(parse_pdb(tmp_path))
            # Same number of chains
            assert set(cg_structure.chains.keys()) == set(reloaded.chains.keys())
            # Atom counts should match
            orig_atoms = sum(len(r.atoms) for c in cg_structure for r in c)
            new_atoms = sum(len(r.atoms) for c in reloaded for r in c)
            assert orig_atoms == new_atoms
        finally:
            os.unlink(tmp_path)

    def test_cif_coordinates_preserved(self, cg_structure):
        """Coordinates survive round-trip within floating point tolerance."""
        with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as f:
            tmp_path = f.name
        try:
            write_cif(cg_structure.to_dict(), tmp_path)
            reloaded = Structure.from_dict(parse_cif(tmp_path))
            orig_coords = cg_structure.get_coords()
            new_coords = reloaded.get_coords()
            np.testing.assert_allclose(orig_coords, new_coords, atol=0.01)
        finally:
            os.unlink(tmp_path)


# ──────────────────────────────────────────────
# 3. Data Structures
# ──────────────────────────────────────────────

class TestDataStructures:
    def test_structure_from_dict(self):
        raw = parse_cif(CG_CIF)
        s = Structure.from_dict(raw)
        assert len(s.chains) > 0

    def test_structure_is_coarse_grained(self, cg_structure):
        assert cg_structure.is_coarse_grained()

    def test_structure_not_coarse_grained(self, allatom_structure):
        assert not allatom_structure.is_coarse_grained()

    def test_coarse_grain_preserves_residue_count(self, allatom_structure):
        cg = allatom_structure.coarse_grain()
        assert cg.is_coarse_grained()
        # Each residue should have exactly one CA
        for chain in cg:
            for res in chain:
                assert len(res.atoms) == 1
                assert res.atoms[0].name == "CA"

    def test_coarse_grain_preserves_chains(self, allatom_structure):
        cg = allatom_structure.coarse_grain()
        assert set(cg.chains.keys()) == set(allatom_structure.chains.keys())

    def test_coarse_grain_does_not_mark_residues_built(self, allatom_structure):
        cg = allatom_structure.coarse_grain()
        for chain in cg:
            for residue in chain:
                assert residue.was_built is False

    def test_coarse_grain_nucleic_acid_residue_uses_com(self):
        s = Structure("mixed")
        s.add_atom("A", "1", "ALA", "N", "N", 0.0, 0.0, 0.0)
        s.add_atom("A", "1", "ALA", "CA", "C", 1.0, 0.0, 0.0)
        s.add_atom("B", "1", "DA", "P", "P", 0.0, 0.0, 0.0)
        s.add_atom("B", "1", "DA", "C1'", "C", 2.0, 0.0, 0.0)
        s.add_atom("B", "1", "DA", "N9", "N", 4.0, 0.0, 0.0)

        cg = s.coarse_grain()

        dna_atom = cg.chains["B"].residues["1"].atoms[0]
        expected_x = (
            ELEMENT_MASSES["P"] * 0.0
            + ELEMENT_MASSES["C"] * 2.0
            + ELEMENT_MASSES["N"] * 4.0
        ) / (ELEMENT_MASSES["P"] + ELEMENT_MASSES["C"] + ELEMENT_MASSES["N"])

        assert set(cg.chains) == {"A", "B"}
        assert dna_atom.name == "CA"
        assert dna_atom.x == pytest.approx(expected_x)
        assert dna_atom.y == pytest.approx(0.0)
        assert dna_atom.z == pytest.approx(0.0)

    def test_get_coords_shape(self, cg_structure):
        coords = cg_structure.get_coords()
        assert coords.ndim == 2
        assert coords.shape[1] == 3

    def test_add_atom(self):
        s = Structure("test")
        s.add_atom("A", "1", "ALA", "CA", "C", 1.0, 2.0, 3.0)
        assert len(s.chains) == 1
        assert "1" in s.chains["A"].residues
        atom = s.chains["A"].residues["1"].atoms[0]
        assert atom.x == 1.0
        assert atom.y == 2.0
        assert atom.z == 3.0
        assert s.chains["A"].residues["1"].was_built is False

    def test_add_atom_can_mark_residue_built_when_requested(self):
        s = Structure("test")
        s.add_atom("A", "1", "ALA", "CA", "C", 1.0, 2.0, 3.0, set_was_built=True)
        assert s.chains["A"].residues["1"].was_built is True

    def test_get_amino_acid_sequence(self, cg_structure):
        seq = cg_structure.chains["A"].get_amino_acid_sequence()
        assert len(seq) > 0
        assert all(c in "ACDEFGHIKLMNPQRSTVWYX" for c in seq)

    def test_to_dict_and_back(self, cg_structure):
        d = cg_structure.to_dict()
        rebuilt = Structure.from_dict(d)
        assert set(rebuilt.chains.keys()) == set(cg_structure.chains.keys())

    def test_translate(self):
        s = Structure("test")
        s.add_atom("A", "1", "ALA", "CA", "C", 0.0, 0.0, 0.0)
        s.translate([10.0, 20.0, 30.0])
        atom = s.chains["A"].residues["1"].atoms[0]
        assert abs(atom.x - 10.0) < 1e-6
        assert abs(atom.y - 20.0) < 1e-6
        assert abs(atom.z - 30.0) < 1e-6

    def test_rotate_identity(self):
        s = Structure("test")
        s.add_atom("A", "1", "ALA", "CA", "C", 5.0, 0.0, 0.0)
        identity = np.eye(3)
        s.rotate(identity, center=np.array([0.0, 0.0, 0.0]))
        atom = s.chains["A"].residues["1"].atoms[0]
        assert abs(atom.x - 5.0) < 1e-6

    def test_merge_structures(self):
        s1 = Structure("s1")
        s1.add_atom("A", "1", "ALA", "CA", "C", 0.0, 0.0, 0.0)
        s2 = Structure("s2")
        s2.add_atom("B", "1", "GLY", "CA", "C", 10.0, 0.0, 0.0)
        s1.merge(s2)
        assert "A" in s1.chains
        assert "B" in s1.chains

    def test_merge_with_chain_rename(self):
        s1 = Structure("s1")
        s1.add_atom("A", "1", "ALA", "CA", "C", 0.0, 0.0, 0.0)
        s2 = Structure("s2")
        s2.add_atom("A", "1", "GLY", "CA", "C", 10.0, 0.0, 0.0)
        s1.merge(s2, rename_chains=True)
        assert len(s1.chains) == 2

    def test_merge_with_chain_rename_does_not_mutate_input(self):
        s1 = Structure("s1")
        s1.add_atom("A", "1", "ALA", "CA", "C", 0.0, 0.0, 0.0)
        s2 = Structure("s2")
        s2.add_atom("A", "1", "GLY", "CA", "C", 10.0, 0.0, 0.0)

        s1.merge(s2, rename_chains=True)

        assert list(s2.chains) == ["A"]
        atom = s2.chains["A"].residues["1"].atoms[0]
        assert atom.data["label_asym_id"] == "A"
        assert atom.data["auth_asym_id"] == "A"

    def test_merge_collision_no_rename_raises(self):
        s1 = Structure("s1")
        s1.add_atom("A", "1", "ALA", "CA", "C", 0.0, 0.0, 0.0)
        s2 = Structure("s2")
        s2.add_atom("A", "1", "GLY", "CA", "C", 10.0, 0.0, 0.0)
        with pytest.raises(ValueError, match="Chain ID"):
            s1.merge(s2, rename_chains=False)

    def test_position_relative_to(self):
        s1 = Structure("s1")
        s1.add_atom("A", "1", "ALA", "CA", "C", 0.0, 0.0, 0.0)
        s2 = Structure("s2")
        s2.add_atom("B", "1", "GLY", "CA", "C", 100.0, 0.0, 0.0)
        s2.position_relative_to(
            s1, self_chain="B", self_res="1",
            other_chain="A", other_res="1",
            target_distance=20.0,
        )
        coord = s2.get_residue_coord("B", "1")
        dist = np.linalg.norm(coord - np.array([0.0, 0.0, 0.0]))
        assert abs(dist - 20.0) < 1.0  # within 1 angstrom tolerance


# ──────────────────────────────────────────────
# 4. Missing Residue Detection
# ──────────────────────────────────────────────

class TestMissingResidues:
    def test_get_missing_by_number_returns_dict(self, cg_structure):
        seq = cg_structure.chains["A"].get_amino_acid_sequence()
        result = get_missing_residues_by_number(CG_CIF, {"A": seq})
        assert isinstance(result, dict)
        assert "A" in result

    def test_complete_sequence_has_no_missing(self, cg_structure):
        """If we pass the exact sequence, nothing should be missing."""
        seq = cg_structure.chains["A"].get_amino_acid_sequence()
        result = get_missing_residues_by_number(CG_CIF, {"A": seq})
        for block in result["A"].values():
            assert block["status"] == "present"

    def test_longer_sequence_detects_missing(self, cg_structure):
        """Prepend residues — should detect N-terminal missing block."""
        seq = cg_structure.chains["A"].get_amino_acid_sequence()
        extended = "MMMM" + seq  # 4 extra at N-term
        result = get_missing_residues_by_number(CG_CIF, {"A": extended})
        missing_blocks = [b for b in result["A"].values() if b["status"] == "missing"]
        assert len(missing_blocks) > 0


# ──────────────────────────────────────────────
# 5. Sequence Alignment
# ──────────────────────────────────────────────

class TestAlignment:
    def test_identical_sequences(self):
        a1, a2 = affine_global_align("ACDEF", "ACDEF")
        assert "".join(a1) == "ACDEF"
        assert "".join(a2) == "ACDEF"

    def test_gap_in_second(self):
        a1, a2 = affine_global_align("ACDEF", "AEF")
        # Reference should be gapped to align with the shorter one
        assert len(a1) == len(a2)

    def test_completely_different(self):
        a1, a2 = affine_global_align("AAAA", "GGGG")
        assert len(a1) == len(a2)


# ──────────────────────────────────────────────
# 6. Build File Validation
# ──────────────────────────────────────────────

class TestBuildFile:
    def test_valid_build_file(self, tmp_path):
        data = {
            "input_path": CG_CIF,
            "output_path": "/some/output.cif",
            "chains_to_modify": [{"chain_id": "A", "sequence": "ACDEF"}],
        }
        path = str(tmp_path / "build.json")
        with open(path, "w") as f:
            json.dump(data, f)
        result = read_build_file(path)
        result = set_up_data(result)
        assert result["bond_length"] == 3.8
        assert result["stiffness_angle"] == 135
        assert result["clash_distance"] == 3.4

    def test_missing_input_path(self, tmp_path):
        data = {
            "output_path": "/some/output.cif",
            "chains_to_modify": [{"chain_id": "A", "sequence": "ACDEF"}],
        }
        path = str(tmp_path / "build.json")
        with open(path, "w") as f:
            json.dump(data, f)
        with pytest.raises(ValueError, match="Missing required"):
            set_up_data(read_build_file(path))

    def test_missing_chain_sequence(self, tmp_path):
        data = {
            "input_path": "/some/path.cif",
            "output_path": "/some/output.cif",
            "chains_to_modify": [{"chain_id": "A"}],
        }
        path = str(tmp_path / "build.json")
        with open(path, "w") as f:
            json.dump(data, f)
        with pytest.raises(ValueError, match="Missing required parameter"):
            set_up_data(read_build_file(path))

    def test_invalid_bond_length(self, tmp_path):
        data = {
            "input_path": "/some/path.cif",
            "output_path": "/some/output.cif",
            "bond_length": -1,
            "chains_to_modify": [{"chain_id": "A", "sequence": "ACDEF"}],
        }
        path = str(tmp_path / "build.json")
        with open(path, "w") as f:
            json.dump(data, f)
        with pytest.raises(ValueError, match="Invalid bond_length"):
            set_up_data(read_build_file(path))

    def test_invalid_stiffness_angle(self, tmp_path):
        data = {
            "input_path": "/some/path.cif",
            "output_path": "/some/output.cif",
            "stiffness_angle": 200,
            "chains_to_modify": [{"chain_id": "A", "sequence": "ACDEF"}],
        }
        path = str(tmp_path / "build.json")
        with open(path, "w") as f:
            json.dump(data, f)
        with pytest.raises(ValueError, match="Invalid stiffness_angle"):
            set_up_data(read_build_file(path))


# ──────────────────────────────────────────────
# 7. Geometry / Modify Structure
# ──────────────────────────────────────────────

class TestGeometry:
    def test_generate_next_calpha_distance(self):
        prev = np.array([0.0, 0.0, 0.0])
        pprev = np.array([-3.8, 0.0, 0.0])
        nxt = generate_next_calpha(prev, pprev, bond_length=3.8, stiffness_angle=120)
        dist = np.linalg.norm(nxt - prev)
        assert abs(dist - 3.8) < 0.01

    def test_sphere_points_distance(self):
        center = np.array([0.0, 0.0, 0.0])
        pts = generate_sphere_points(center, radius=5.0, num_points=100)
        dists = np.linalg.norm(pts - center, axis=1)
        np.testing.assert_allclose(dists, 5.0, atol=0.01)

    def test_get_centroid(self):
        pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        c = get_centroid(pts)
        expected = np.array([2.0 / 3, 2.0 / 3, 0.0])
        np.testing.assert_allclose(c, expected, atol=1e-6)

    def test_get_neighbors_in_sphere(self):
        center = np.array([0.0, 0.0, 0.0])
        coords = np.array([
            [1.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ])
        neighbors = get_neighbors_in_sphere(center, coords, radius=3.0)
        assert len(neighbors) == 2  # first and third are within 3.0

    def test_non_clashing_filters_close(self):
        candidates = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        obstacles = np.array([[0.5, 0.0, 0.0]])
        result = get_non_clashing_coords(candidates, obstacles, min_distance=3.0)
        assert len(result) == 1
        assert result[0][0] == 10.0

    def test_rotation_matrix_identity(self):
        v = np.array([1.0, 0.0, 0.0])
        R = rotation_matrix_from_vectors(v, v)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-6)

    def test_build_loop_coordinates_final_residue_stays_one_bond_from_end(self):
        np.random.seed(0)
        start = np.array([0.0, 0.0, 0.0])
        end = np.array([7.6, 0.0, 0.0])
        obstacles = np.array([
            [-20.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ])

        coords = build_idr.build_loop_coordinates(
            starting_coordinate=start,
            ending_coordinate=end,
            all_current_coordinates=obstacles,
            num_residues=1,
            start_index=10,
            bond_length=3.8,
            clash_distance=1.0,
            show_progress=False,
        )

        assert coords.shape == (1, 3)
        assert np.linalg.norm(coords[0] - start) == pytest.approx(3.8, abs=0.05)
        assert np.linalg.norm(coords[0] - end) == pytest.approx(3.8, abs=0.05)

    def test_build_loop_coordinates_falls_back_when_greedy_schedule_fails(self, monkeypatch):
        np.random.seed(0)
        start = np.array([0.0, 0.0, 0.0])
        end = np.array([11.4, 0.0, 0.0])
        obstacles = np.array([
            [-20.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ])

        def fail_greedy(*args, **kwargs):
            raise ValueError("forced greedy failure")

        monkeypatch.setattr(build_idr, "_build_loop_coordinates_greedy", fail_greedy)

        coords = build_idr.build_loop_coordinates(
            starting_coordinate=start,
            ending_coordinate=end,
            all_current_coordinates=obstacles,
            num_residues=2,
            start_index=10,
            bond_length=3.8,
            clash_distance=1.0,
            show_progress=False,
        )

        assert coords.shape == (2, 3)
        assert np.linalg.norm(coords[0] - start) == pytest.approx(3.8, abs=0.05)
        assert np.linalg.norm(coords[1] - coords[0]) == pytest.approx(3.8, abs=0.05)
        assert np.linalg.norm(end - coords[-1]) == pytest.approx(3.8, abs=0.05)

    def test_generate_directed_path_uses_bidirectional_fallback_when_beam_stalls(self, monkeypatch):
        start = np.array([0.0, 0.0, 0.0])
        end = np.array([11.4, 0.0, 0.0])
        expected = np.array([
            [3.8, 0.0, 0.0],
            [7.6, 0.0, 0.0],
        ])

        monkeypatch.setattr(
            experimental,
            "_generate_step_candidates",
            lambda *args, **kwargs: np.empty((0, 3)),
        )
        monkeypatch.setattr(
            experimental,
            "_generate_bidirectional_path",
            lambda *args, **kwargs: expected,
        )

        coords = experimental.generate_directed_path(
            tuple(start),
            tuple(end),
            existing_coords=np.empty((0, 3)),
            num_coords=2,
            target_dist=3.8,
            clash_dist=1.0,
            num_sphere_points=64,
        )

        np.testing.assert_allclose(coords, expected)


# ──────────────────────────────────────────────
# 8. Protein Math
# ──────────────────────────────────────────────

class TestProteinMath:
    def test_calculate_distance(self):
        d = calculate_distance(np.array([0, 0, 0]), np.array([3, 4, 0]))
        assert abs(d - 5.0) < 1e-6

    def test_find_points_within_sphere(self):
        pts = np.array([[0, 0, 0], [1, 0, 0], [10, 0, 0]])
        center = np.array([0, 0, 0])
        inside = find_points_within_sphere(pts, center, 5.0)
        assert len(inside) == 2

    def test_distance_zero_same_point(self):
        p = np.array([1.0, 2.0, 3.0])
        assert calculate_distance(p, p) == 0.0


# ──────────────────────────────────────────────
# 9. Verify non-clashing
# ──────────────────────────────────────────────

class TestClashDetection:
    def test_non_clashing_structure(self, cg_structure):
        """The provided CG structure should not have clashing atoms at 2.0 Å."""
        # Use a generous threshold since this is CG
        assert cg_structure.verify_non_clashing(min_distance=2.0)

    def test_clashing_atoms_detected(self):
        s = Structure("test")
        s.add_atom("A", "1", "ALA", "CA", "C", 0.0, 0.0, 0.0)
        s.add_atom("A", "2", "GLY", "CA", "C", 0.5, 0.0, 0.0)
        assert not s.verify_non_clashing(min_distance=1.0)


class TestValidation:
    def test_validate_bond_lengths_normal(self, cg_structure):
        """CG structure should have mostly reasonable CA-CA bonds."""
        issues = cg_structure.validate_bond_lengths(expected=3.8, tolerance=1.5)
        # Some chain breaks are expected in a real structure, but most bonds should be fine
        assert isinstance(issues, list)

    def test_validate_bond_lengths_flags_bad(self):
        """Two atoms 20 Å apart should be flagged."""
        s = Structure("test")
        s.add_atom("A", "1", "ALA", "CA", "C", 0.0, 0.0, 0.0)
        s.add_atom("A", "2", "GLY", "CA", "C", 20.0, 0.0, 0.0)
        issues = s.validate_bond_lengths(expected=3.8, tolerance=1.0)
        assert len(issues) == 1
        assert issues[0]['distance'] == 20.0

    def test_detect_chain_breaks(self):
        """Detect a break when residues are far apart."""
        s = Structure("test")
        s.add_atom("A", "1", "ALA", "CA", "C", 0.0, 0.0, 0.0)
        s.add_atom("A", "2", "GLY", "CA", "C", 3.8, 0.0, 0.0)
        s.add_atom("A", "3", "VAL", "CA", "C", 50.0, 0.0, 0.0)
        breaks = s.detect_chain_breaks(max_distance=5.0)
        assert len(breaks) == 1
        assert breaks[0]['res_before'] == '2'
        assert breaks[0]['res_after'] == '3'

    def test_no_chain_breaks(self):
        s = Structure("test")
        s.add_atom("A", "1", "ALA", "CA", "C", 0.0, 0.0, 0.0)
        s.add_atom("A", "2", "GLY", "CA", "C", 3.8, 0.0, 0.0)
        breaks = s.detect_chain_breaks(max_distance=5.0)
        assert len(breaks) == 0

    def test_check_sequence_consistency_match(self):
        s = Structure("test")
        s.add_atom("A", "1", "ALA", "CA", "C", 0.0, 0.0, 0.0)
        s.add_atom("A", "2", "GLY", "CA", "C", 3.8, 0.0, 0.0)
        result = s.check_sequence_consistency({"A": "AG"})
        assert result["A"]["matches"] is True

    def test_check_sequence_consistency_mismatch(self):
        s = Structure("test")
        s.add_atom("A", "1", "ALA", "CA", "C", 0.0, 0.0, 0.0)
        s.add_atom("A", "2", "GLY", "CA", "C", 3.8, 0.0, 0.0)
        result = s.check_sequence_consistency({"A": "GG"})
        assert result["A"]["matches"] is False

    def test_check_sequence_missing_chain(self):
        s = Structure("test")
        s.add_atom("A", "1", "ALA", "CA", "C", 0.0, 0.0, 0.0)
        result = s.check_sequence_consistency({"B": "M"})
        assert result["B"]["matches"] is False
        assert result["B"]["structure_seq"] == ''


class TestNonstandardResidues:
    def test_selenomethionine_maps_to_M(self):
        from protein_modifier.data.amino_acids import NONSTANDARD_AA_MAP_3_TO_1
        assert NONSTANDARD_AA_MAP_3_TO_1['MSE'] == 'M'

    def test_phosphoserine_maps_to_S(self):
        from protein_modifier.data.amino_acids import NONSTANDARD_AA_MAP_3_TO_1
        assert NONSTANDARD_AA_MAP_3_TO_1['SEP'] == 'S'

    def test_chain_sequence_uses_nonstandard(self):
        """A chain with MSE should report 'M' in its sequence."""
        s = Structure("test")
        s.add_atom("A", "1", "ALA", "CA", "C", 0.0, 0.0, 0.0)
        s.add_atom("A", "2", "MSE", "CA", "C", 3.8, 0.0, 0.0)
        seq = s.chains["A"].get_amino_acid_sequence()
        assert seq == "AM"


class TestRetryParameterization:
    def test_attempts_default_in_build_data(self):
        """set_up_data should supply default attempts."""
        data = {
            "input_path": CG_CIF,
            "output_path": "out.cif",
            "chains_to_modify": [{"chain_id": "A", "sequence": "ACDEF"}],
        }
        result = set_up_data(data)
        assert result['attempts'] == 5

    def test_replicates_default_in_build_data(self):
        """set_up_data should supply default replicates."""
        data = {
            "input_path": CG_CIF,
            "output_path": "out.cif",
            "chains_to_modify": [{"chain_id": "A", "sequence": "ACDEF"}],
        }
        result = set_up_data(data)
        assert result['replicates'] == 1

    def test_attempts_custom_value(self):
        data = {
            "input_path": CG_CIF,
            "output_path": "out.cif",
            "chains_to_modify": [{"chain_id": "A", "sequence": "ACDEF"}],
            "attempts": 10,
        }
        result = set_up_data(data)
        assert result['attempts'] == 10

    def test_attempts_invalid_raises(self):
        data = {
            "input_path": "test.cif",
            "output_path": "out.cif",
            "chains_to_modify": [{"chain_id": "A", "sequence": "ACDEF"}],
            "attempts": 0,
        }
        with pytest.raises(ValueError, match="Invalid attempts"):
            set_up_data(data)


# ──────────────────────────────────────────────
# 12. End-to-End Integration Test
# ──────────────────────────────────────────────

class TestIntegration:
    def test_modify_protein_builds_c_term(self, tmp_path):
        """Full modify_protein workflow: add 3 C-terminal residues to chain R."""
        from protein_modifier import modify_protein
        # Chain R existing sequence (31 residues, ids -1 to 29)
        existing_seq = "SMDAIKKKMQMLKLDKENALDRAEQAEADKA"
        extended_seq = existing_seq + "AKE"  # add 3 C-terminal residues
        build_data = {
            "input_path": CG_CIF,
            "output_path": str(tmp_path / "output.cif"),
            "chains_to_modify": [
                {"chain_id": "R", "sequence": extended_seq},
            ],
            "attempts": 10,
        }
        build_path = str(tmp_path / "build.json")
        with open(build_path, "w") as f:
            json.dump(build_data, f)
        modify_protein(build_path)
        # Output CIF should exist
        assert os.path.exists(str(tmp_path / "output.cif"))
        # Build report should exist
        assert os.path.exists(str(tmp_path / "output_build_report.txt"))
        # Reload and verify chain R has more residues than original
        result = Structure.from_dict(parse_cif(str(tmp_path / "output.cif")))
        assert "R" in result.chains
        assert len(result.chains["R"].residues) > len(existing_seq)

    def test_modify_protein_preserves_nucleic_acids_during_coarse_graining(self, tmp_path):
        from protein_modifier import modify_protein

        structure = Structure("mixed")
        structure.add_atom("A", "1", "ALA", "N", "N", 0.0, 0.0, 0.0)
        structure.add_atom("A", "1", "ALA", "CA", "C", 1.0, 0.0, 0.0)
        structure.add_atom("A", "1", "C", "C", "C", 2.0, 0.0, 0.0)
        structure.add_atom("B", "1", "DA", "P", "P", 10.0, 0.0, 0.0)
        structure.add_atom("B", "1", "C1'", "C", "C", 12.0, 0.0, 0.0)
        structure.add_atom("B", "1", "N9", "N", "N", 14.0, 0.0, 0.0)
        structure.add_atom("B", "2", "DT", "P", "P", 20.0, 1.0, 0.0)
        structure.add_atom("B", "2", "C1'", "C", "C", 22.0, 1.0, 0.0)
        structure.add_atom("B", "2", "N1", "N", "N", 24.0, 1.0, 0.0)

        input_path = str(tmp_path / "mixed_input.cif")
        output_path = str(tmp_path / "mixed_output.cif")
        build_path = str(tmp_path / "build.json")
        write_cif(structure.to_dict(), input_path)

        with open(build_path, "w") as f:
            json.dump(
                {
                    "input_path": input_path,
                    "output_path": output_path,
                    "chains_to_modify": [{"chain_id": "A", "sequence": "A"}],
                },
                f,
            )

        modify_protein(build_path)

        result = Structure.from_dict(parse_cif(output_path))
        assert set(result.chains) == {"A", "B"}
        assert len(result.chains["A"].residues["1"].atoms) == 1
        assert len(result.chains["B"].residues) == 2
        for residue in result.chains["B"]:
            assert len(residue.atoms) == 1
            assert residue.atoms[0].name == "CA"

        expected_x = (
            ELEMENT_MASSES["P"] * 10.0
            + ELEMENT_MASSES["C"] * 12.0
            + ELEMENT_MASSES["N"] * 14.0
        ) / (ELEMENT_MASSES["P"] + ELEMENT_MASSES["C"] + ELEMENT_MASSES["N"])
        assert result.chains["B"].residues["1"].atoms[0].x == pytest.approx(expected_x, abs=1e-3)

    def test_modify_protein_raises_when_output_directory_exists(self, tmp_path):
        from protein_modifier import modify_protein

        existing_output_dir = tmp_path / "outputs" / "replicate_1"
        existing_output_dir.mkdir(parents=True)
        build_data = {
            "input_path": CG_CIF,
            "output_path": str(tmp_path / "outputs" / "output.cif"),
            "chains_to_modify": [
                {"chain_id": "R", "sequence": "SMDAIKKKMQMLKLDKENALDRAEQAEADKA"},
            ],
            "replicates": 2,
        }
        build_path = tmp_path / "build.json"
        with open(build_path, "w") as f:
            json.dump(build_data, f)

        with pytest.raises(FileExistsError, match="Output target already exists"):
            modify_protein(str(build_path))

    def test_modify_protein_overwrite_allows_existing_output_directory(self, tmp_path):
        from protein_modifier import modify_protein

        existing_output_dir = tmp_path / "outputs" / "replicate_1"
        existing_output_dir.mkdir(parents=True)
        build_data = {
            "input_path": CG_CIF,
            "output_path": str(tmp_path / "outputs" / "output.cif"),
            "chains_to_modify": [
                {"chain_id": "R", "sequence": "SMDAIKKKMQMLKLDKENALDRAEQAEADKA"},
            ],
            "replicates": 2,
        }
        build_path = tmp_path / "build.json"
        with open(build_path, "w") as f:
            json.dump(build_data, f)

        modify_protein(str(build_path), overwrite=True)

        assert (tmp_path / "outputs" / "replicate_1" / "output.cif").exists()
        assert (tmp_path / "outputs" / "replicate_2" / "output.cif").exists()


class TestBuildCLI:
    def test_build_cli_passes_overwrite_flag(self, monkeypatch, tmp_path):
        build_path = tmp_path / "build.json"
        build_path.write_text("{}")

        captured = {}

        def fake_modify_protein(build_file, coarse_grain=True, overwrite=False):
            captured["build_file"] = build_file
            captured["coarse_grain"] = coarse_grain
            captured["overwrite"] = overwrite

        monkeypatch.setattr(cli, "modify_protein", fake_modify_protein)

        with mock.patch.object(sys, "argv", ["protein-modifier", "build", str(build_path), "--overwrite"]):
            cli.main()

        assert captured == {
            "build_file": str(build_path),
            "coarse_grain": True,
            "overwrite": True,
        }


# ──────────────────────────────────────────────
# 13. Extended Coverage
# ──────────────────────────────────────────────

class TestExtendedCoverage:
    def test_assign_bead_type_maps_nucleic_acids_to_nucleotide_indices(self, tmp_path, monkeypatch):
        structure = Structure("mixed")
        structure.add_atom("A", "1", "ALA", "CA", "C", 0.0, 0.0, 0.0)
        structure.add_atom("B", "1", "DA", "CA", "C", 1.0, 0.0, 0.0)
        structure.add_atom("B", "2", "DT", "CA", "C", 2.0, 0.0, 0.0)

        def fake_sasa(current_structure, structure_file, probe_radius=1.4):
            current_structure.chains["A"].residues["1"].assign_solvent_accessibility(1)
            current_structure.chains["B"].residues["1"].assign_solvent_accessibility(1)
            current_structure.chains["B"].residues["2"].assign_solvent_accessibility(1)
            return current_structure

        monkeypatch.setattr(sim_file_generation, "get_sasa_by_residue", fake_sasa)

        input_path = tmp_path / "mixed.cif"
        write_cif(structure.to_dict(), str(input_path))
        typed = sim_file_generation.assign_bead_type(structure, str(input_path))

        assert typed.chains["A"].residues["1"].bead_type == 26
        assert typed.chains["B"].residues["1"].bead_type == 41
        assert typed.chains["B"].residues["2"].bead_type == 44

    def test_assign_bead_type_does_not_apply_buried_offset_to_canonical_nucleic_acids(self, tmp_path, monkeypatch):
        structure = Structure("rna")
        structure.add_atom("A", "1", "RPA", "CA", "C", 0.0, 0.0, 0.0)

        def fake_sasa(current_structure, structure_file, probe_radius=1.4):
            current_structure.chains["A"].residues["1"].assign_solvent_accessibility(1)
            return current_structure

        monkeypatch.setattr(sim_file_generation, "get_sasa_by_residue", fake_sasa)

        input_path = tmp_path / "rna.cif"
        write_cif(structure.to_dict(), str(input_path))
        typed = sim_file_generation.assign_bead_type(structure, str(input_path))

        assert typed.chains["A"].residues["1"].bead_type == 41

    def test_write_seq_dat_uses_nucleotide_bead_types(self, tmp_path, monkeypatch):
        structure = Structure("mixed")
        structure.add_atom("A", "1", "ALA", "CA", "C", 0.0, 0.0, 0.0)
        structure.add_atom("B", "1", "DA", "CA", "C", 10.0, 0.0, 0.0)
        structure.add_atom("B", "2", "DT", "CA", "C", 20.0, 0.0, 0.0)

        def fake_sasa(current_structure, structure_file, probe_radius=1.4):
            current_structure.chains["A"].residues["1"].assign_solvent_accessibility(1)
            current_structure.chains["B"].residues["1"].assign_solvent_accessibility(1)
            current_structure.chains["B"].residues["2"].assign_solvent_accessibility(0)
            return current_structure

        monkeypatch.setattr(sim_file_generation, "get_sasa_by_residue", fake_sasa)

        input_path = tmp_path / "mixed.cif"
        output_path = tmp_path / "sequence.dat"
        write_cif(structure.to_dict(), str(input_path))

        sim_file_generation.write_seq_dat(str(input_path), str(output_path))

        with open(output_path) as handle:
            atom_lines = []
            in_atoms = False
            for line in handle:
                stripped = line.strip()
                if stripped == "Atoms":
                    in_atoms = True
                    continue
                if not in_atoms or not stripped:
                    continue
                if stripped == "Bonds":
                    break
                atom_lines.append(stripped)

        bead_types = [int(line.split()[2]) for line in atom_lines]
        assert bead_types == [26, 41, 44]

    def test_write_seq_dat_uses_emitted_atom_ids_for_bonds(self, tmp_path, monkeypatch):
        structure = Structure("protein")
        structure.add_atom("A", "1", "ALA", "CA", "C", 0.0, 0.0, 0.0)
        structure.add_atom("A", "2", "GLY", "CA", "C", 3.8, 0.0, 0.0)
        structure.add_atom("A", "3", "SER", "CA", "C", 7.6, 0.0, 0.0)
        raw_structure = structure.to_dict()
        raw_structure["A"]["1"][0]["id"] = "10"
        raw_structure["A"]["2"][0]["id"] = "20"
        raw_structure["A"]["3"][0]["id"] = "30"

        def fake_sasa(current_structure, structure_file, probe_radius=1.4):
            for residue_id in ("1", "2", "3"):
                current_structure.chains["A"].residues[residue_id].assign_solvent_accessibility(0)
            return current_structure

        monkeypatch.setattr(sim_file_generation, "parse_structure", lambda _: raw_structure)
        monkeypatch.setattr(sim_file_generation, "get_sasa_by_residue", fake_sasa)

        output_path = tmp_path / "sequence.dat"
        sim_file_generation.write_seq_dat(str(tmp_path / "input.cif"), str(output_path))

        with open(output_path) as handle:
            bond_lines = []
            in_bonds = False
            for line in handle:
                stripped = line.strip()
                if stripped == "Bonds":
                    in_bonds = True
                    continue
                if not in_bonds or not stripped:
                    continue
                bond_lines.append(stripped)

        assert bond_lines == ["1 1 1 2", "2 1 2 3"]

    def test_get_lammps_group_numbers_returns_frozen_ranges(self, tmp_path):
        input_structure = Structure("input")
        input_structure.add_atom("A", "1", "ALA", "CA", "C", 0.0, 0.0, 0.0)
        input_structure.add_atom("A", "4", "GLU", "CA", "C", 4.0, 0.0, 0.0)
        input_structure.add_atom("A", "5", "PHE", "CA", "C", 5.0, 0.0, 0.0)
        input_structure.add_atom("B", "1", "GLY", "CA", "C", 0.0, 5.0, 0.0)
        input_structure.add_atom("B", "2", "HIS", "CA", "C", 1.0, 5.0, 0.0)

        final_structure = Structure("final")
        final_structure.add_atom("A", "1", "ALA", "CA", "C", 0.0, 0.0, 0.0)
        final_structure.add_atom("A", "2", "CYS", "CA", "C", 1.0, 0.0, 0.0)
        final_structure.add_atom("A", "3", "ASP", "CA", "C", 2.0, 0.0, 0.0)
        final_structure.add_atom("A", "4", "GLU", "CA", "C", 3.0, 0.0, 0.0)
        final_structure.add_atom("A", "5", "PHE", "CA", "C", 4.0, 0.0, 0.0)
        final_structure.add_atom("B", "1", "GLY", "CA", "C", 0.0, 5.0, 0.0)
        final_structure.add_atom("B", "2", "HIS", "CA", "C", 1.0, 5.0, 0.0)

        input_path = tmp_path / "input.cif"
        final_path = tmp_path / "final.cif"
        write_cif(input_structure.to_dict(), str(input_path))
        write_cif(final_structure.to_dict(), str(final_path))

        build_data = {
            "input_path": str(input_path),
            "output_path": str(tmp_path / "output.cif"),
            "chains_to_modify": [{"chain_id": "A", "sequence": "ACDEF"}],
        }
        build_path = tmp_path / "build.json"
        with open(build_path, "w") as f:
            json.dump(build_data, f)

        ranges = get_lammps_group_numbers(
            structure_file_path=str(final_path),
            output_path=str(tmp_path / "unused.in"),
            json_input_path=str(build_path),
            boxdims=800,
        )

        assert ranges == "1:1 4:7"

    def test_centroid_returns_ndarray(self):
        """get_centroid should return an np.ndarray, not a tuple."""
        coords = [(1.0, 2.0, 3.0), (5.0, 6.0, 7.0)]
        centroid = get_centroid(coords)
        assert isinstance(centroid, np.ndarray)
        np.testing.assert_allclose(centroid, [3.0, 4.0, 5.0])

    def test_centroid_empty_list(self):
        """get_centroid on empty list returns None."""
        assert get_centroid([]) is None

    def test_rotation_matrix_non_identity(self):
        """Rotation matrix between non-parallel vectors should not be identity."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        R = rotation_matrix_from_vectors(v1, v2)
        result = R @ v1
        np.testing.assert_allclose(result, v2, atol=1e-10)

    def test_parse_build_file_duplicate_chain(self, tmp_path):
        """set_up_data rejects duplicate chain_ids."""
        data = {
            "input_path": CG_CIF,
            "output_path": "out.cif",
            "chains_to_modify": [
                {"chain_id": "A", "sequence": "ACDEF"},
                {"chain_id": "A", "sequence": "GGGGG"},
            ],
        }
        with pytest.raises(ValueError, match="Duplicate chain_id"):
            set_up_data(data)

    def test_parse_build_file_invalid_sequence(self, tmp_path):
        """set_up_data rejects sequences with invalid amino acid codes."""
        data = {
            "input_path": CG_CIF,
            "output_path": "out.cif",
            "chains_to_modify": [
                {"chain_id": "A", "sequence": "ACXEF"},
            ],
        }
        with pytest.raises(ValueError, match="Invalid amino acid code"):
            set_up_data(data)

    def test_parse_build_file_file_not_found(self):
        """set_up_data raises FileNotFoundError for missing input file."""
        data = {
            "input_path": "/nonexistent/path.cif",
            "output_path": "out.cif",
            "chains_to_modify": [
                {"chain_id": "A", "sequence": "ACDEF"},
            ],
        }
        with pytest.raises(FileNotFoundError, match="not found"):
            set_up_data(data)

    def test_chain_breaks_on_cg_structure(self, cg_structure):
        """Real CG structure chain break detection should return a list."""
        breaks = cg_structure.detect_chain_breaks(max_distance=5.0)
        assert isinstance(breaks, list)

    def test_sequence_consistency_cg_structure(self, cg_structure):
        """check_sequence_consistency should work on real CG chains."""
        seq_a = cg_structure.chains["A"].get_amino_acid_sequence()
        result = cg_structure.check_sequence_consistency({"A": seq_a})
        assert result["A"]["matches"] is True

    def test_structure_merge(self, cg_structure):
        """Merge two structures and verify chain count increases."""
        s2 = Structure("extra")
        s2.add_atom("Z", "1", "ALA", "CA", "C", 999.0, 999.0, 999.0)
        original_chains = len(cg_structure.chains)
        merged = cg_structure.merge(s2)
        assert len(merged.chains) == original_chains + 1
        assert "Z" in merged.chains

    @pytest.mark.parametrize("output_ext", [".pdb", ".cif"])
    def test_combine_structures_mixed_formats(self, output_ext):
        first = Structure("first")
        first.add_atom("A", "1", "ALA", "N", "N", 0.0, 0.0, 0.0)
        first.add_atom("A", "1", "ALA", "CA", "C", 1.0, 0.0, 0.0)

        second = Structure("second")
        second.add_atom("A", "1", "GLY", "N", "N", 10.0, 0.0, 0.0)
        second.add_atom("A", "1", "GLY", "CA", "C", 11.0, 0.0, 0.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            first_path = os.path.join(tmpdir, "first.pdb")
            second_path = os.path.join(tmpdir, "second.cif")
            output_path = os.path.join(tmpdir, f"combined{output_ext}")

            write_pdb(first.to_dict(), first_path)
            write_cif(second.to_dict(), second_path)

            result = combine_structures(first_path, second_path, output_path)
            parsed = parse_structure(output_path)

            assert result["n_chains"] == 2
            assert result["n_atoms"] == 4
            assert result["chains"] == ["A", "B"]
            assert sorted(parsed.keys()) == ["A", "B"]

            if output_ext == ".pdb":
                with open(output_path) as handle:
                    serials = [
                        int(line[6:11])
                        for line in handle
                        if line.startswith(("ATOM", "HETATM", "TER"))
                    ]
                assert serials == list(range(1, len(serials) + 1))
            else:
                ids = [
                    int(atom["id"])
                    for chain in parsed.values()
                    for residue in chain.values()
                    for atom in residue
                ]
                assert ids == [1, 2, 3, 4]

    def test_structure_translate(self):
        """Translate a structure and verify coordinates shift."""
        s = Structure("test")
        s.add_atom("A", "1", "ALA", "CA", "C", 0.0, 0.0, 0.0)
        s.translate([10.0, 20.0, 30.0])
        atom = s.chains["A"].residues["1"].atoms[0]
        assert atom.x == 10.0
        assert atom.y == 20.0
        assert atom.z == 30.0


# ──────────────────────────────────────────────
# 14. Structural Alignment (Kabsch)
# ──────────────────────────────────────────────

def _make_linear_structure(coords, chain_id='A', start_res=1):
    """Helper: build a CA-only Structure from an (N,3) array."""
    s = Structure('test')
    for i, (x, y, z) in enumerate(coords):
        s.add_atom(chain_id, str(start_res + i), 'ALA', 'CA', 'C',
                   float(x), float(y), float(z))
    return s


class TestKabschAlign:
    """Low-level tests for the kabsch_align function."""

    def test_identity_alignment(self):
        """Identical point clouds should give identity rotation and zero RMSD."""
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        R, mc, tc = kabsch_align(pts, pts.copy())
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(mc, tc, atol=1e-12)

    def test_pure_translation(self):
        """Translated cloud should align back with identity rotation."""
        pts = np.array([[0, 0, 0], [3, 0, 0], [0, 4, 0], [0, 0, 5]], dtype=float)
        shifted = pts + np.array([10.0, -20.0, 30.0])
        R, mc, tc = kabsch_align(shifted, pts)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)
        aligned = (shifted - mc) @ R.T + tc
        np.testing.assert_allclose(aligned, pts, atol=1e-10)

    def test_known_90deg_rotation(self):
        """90° rotation around Z-axis should be recovered."""
        pts = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], dtype=float)
        # Rotate 90° around Z: (x,y,z) -> (-y,x,z)
        rotated = np.array([[-0, 1, 0], [-1, 0, 0], [0, -1, 0], [1, 0, 0]], dtype=float)
        R, mc, tc = kabsch_align(rotated, pts)
        aligned = (rotated - mc) @ R.T + tc
        np.testing.assert_allclose(aligned, pts, atol=1e-10)

    def test_rotation_plus_translation(self):
        """Combined rotation + translation should be recovered."""
        np.random.seed(42)
        pts = np.random.randn(20, 3) * 10
        # Apply known rotation (180° around X) and translation
        R_true = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
        t_true = np.array([5.0, -3.0, 8.0])
        mobile = pts @ R_true.T + t_true
        R, mc, tc = kabsch_align(mobile, pts)
        aligned = (mobile - mc) @ R.T + tc
        np.testing.assert_allclose(aligned, pts, atol=1e-10)

    def test_proper_rotation_no_reflection(self):
        """Kabsch must return a proper rotation (det = +1), not a reflection."""
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        reflected = pts.copy()
        reflected[:, 2] *= -1  # Mirror Z
        R, _, _ = kabsch_align(reflected, pts)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_too_few_points_raises(self):
        """Fewer than 3 points should raise ValueError."""
        pts = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        with pytest.raises(ValueError, match='At least 3'):
            kabsch_align(pts, pts)

    def test_shape_mismatch_raises(self):
        """Mismatched shapes should raise ValueError."""
        a = np.zeros((5, 3))
        b = np.zeros((4, 3))
        with pytest.raises(ValueError, match='same shape'):
            kabsch_align(a, b)

    def test_rmsd_is_zero_for_identical(self):
        """RMSD after aligning identical sets should be ~0."""
        pts = np.array([[0, 0, 0], [5, 0, 0], [0, 5, 0], [0, 0, 5]], dtype=float)
        R, mc, tc = kabsch_align(pts, pts)
        aligned = (pts - mc) @ R.T + tc
        diff = aligned - pts
        rmsd = np.sqrt((diff ** 2).sum(axis=1).mean())
        assert rmsd < 1e-12

    def test_large_random_cloud(self):
        """Stress test: 10 000 points should align in well under a second."""
        np.random.seed(99)
        pts = np.random.randn(10000, 3) * 50
        angle = np.pi / 3
        Rz = np.array([[np.cos(angle), -np.sin(angle), 0],
                       [np.sin(angle), np.cos(angle), 0],
                       [0, 0, 1]])
        mobile = pts @ Rz.T + np.array([100, -200, 300])
        R, mc, tc = kabsch_align(mobile, pts)
        aligned = (mobile - mc) @ R.T + tc
        np.testing.assert_allclose(aligned, pts, atol=1e-8)


class TestStructureAlignTo:
    """Tests for Structure.align_to() method."""

    def test_align_identical_structures(self, cg_structure):
        """Aligning a structure to itself should give RMSD ≈ 0."""
        import copy
        target = copy.deepcopy(cg_structure)
        result = cg_structure.align_to(target)
        assert result['rmsd'] < 1e-10
        assert result['n_matched'] > 0

    def test_align_translated_structure(self):
        """Translated structure should align back with RMSD ≈ 0."""
        coords = np.array([[0, 0, 0], [3.8, 0, 0], [7.6, 0, 0],
                           [7.6, 3.8, 0], [7.6, 7.6, 0]], dtype=float)
        target = _make_linear_structure(coords)
        mobile = _make_linear_structure(coords + [100, -50, 200])
        result = mobile.align_to(target)
        assert result['rmsd'] < 1e-10
        assert result['n_matched'] == 5

    def test_align_rotated_structure(self):
        """Rotated structure should align back with RMSD ≈ 0."""
        coords = np.array([[0, 0, 0], [3.8, 0, 0], [3.8, 3.8, 0],
                           [0, 3.8, 0], [0, 3.8, 3.8]], dtype=float)
        target = _make_linear_structure(coords)
        # 90° around Z
        Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        mobile = _make_linear_structure(coords @ Rz.T)
        result = mobile.align_to(target)
        assert result['rmsd'] < 1e-10

    def test_align_with_missing_residues(self):
        """Alignment should work when the mobile structure has fewer residues."""
        # Target has 6 residues
        target_coords = np.array([[0, 0, 0], [3.8, 0, 0], [7.6, 0, 0],
                                  [7.6, 3.8, 0], [7.6, 7.6, 0], [3.8, 7.6, 0]],
                                 dtype=float)
        target = _make_linear_structure(target_coords)
        # Mobile has residues 1,2,3,5,6 (missing residue 4)
        mobile_coords = np.array([[0, 0, 0], [3.8, 0, 0], [7.6, 0, 0],
                                  [7.6, 7.6, 0], [3.8, 7.6, 0]], dtype=float)
        mobile = Structure('mobile')
        for i, (rid, (x, y, z)) in enumerate(zip(['1', '2', '3', '5', '6'], mobile_coords)):
            mobile.add_atom('A', rid, 'ALA', 'CA', 'C', float(x), float(y), float(z))
        # Translate mobile far away
        mobile.translate([500.0, 500.0, 500.0])
        result = mobile.align_to(target)
        assert result['n_matched'] == 5  # only 5 shared residues
        assert result['rmsd'] < 1e-10

    def test_align_with_chain_map(self):
        """chain_map allows aligning chains with different IDs."""
        coords = np.array([[0, 0, 0], [3.8, 0, 0], [0, 3.8, 0],
                           [0, 0, 3.8]], dtype=float)
        target = _make_linear_structure(coords, chain_id='X')
        mobile = _make_linear_structure(coords + [50, 50, 50], chain_id='A')
        result = mobile.align_to(target, chain_map={'A': 'X'})
        assert result['rmsd'] < 1e-10
        assert result['n_matched'] == 4

    def test_align_modifies_inplace(self):
        """align_to should modify the mobile structure's coordinates in place."""
        coords = np.array([[0, 0, 0], [3.8, 0, 0], [0, 3.8, 0]], dtype=float)
        target = _make_linear_structure(coords)
        mobile = _make_linear_structure(coords + [100, 0, 0])
        mobile.align_to(target)
        # After alignment, mobile coords should match target
        for res_id in ['1', '2', '3']:
            m_atom = mobile.chains['A'].residues[res_id].atoms[0]
            t_atom = target.chains['A'].residues[res_id].atoms[0]
            np.testing.assert_allclose(
                [m_atom.x, m_atom.y, m_atom.z],
                [t_atom.x, t_atom.y, t_atom.z],
                atol=1e-10,
            )

    def test_align_returns_rotation_matrix(self):
        """The returned rotation matrix should be a proper rotation."""
        coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        target = _make_linear_structure(coords)
        mobile = _make_linear_structure(coords + [10, 20, 30])
        result = mobile.align_to(target)
        R = result['rotation']
        assert R.shape == (3, 3)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_align_too_few_matches_raises(self):
        """Should raise ValueError when fewer than 3 atoms match."""
        s1 = Structure('s1')
        s1.add_atom('A', '1', 'ALA', 'CA', 'C', 0, 0, 0)
        s1.add_atom('A', '2', 'ALA', 'CA', 'C', 1, 0, 0)
        s2 = Structure('s2')
        s2.add_atom('A', '1', 'ALA', 'CA', 'C', 0, 0, 0)
        s2.add_atom('A', '2', 'ALA', 'CA', 'C', 1, 0, 0)
        with pytest.raises(ValueError, match='at least 3'):
            s1.align_to(s2)

    def test_align_no_overlap_raises(self):
        """Completely non-overlapping chain IDs should raise ValueError."""
        s1 = _make_linear_structure(np.eye(3) * 5, chain_id='A')
        s2 = _make_linear_structure(np.eye(3) * 5, chain_id='Z')
        with pytest.raises(ValueError, match='at least 3'):
            s1.align_to(s2)

    def test_align_multi_chain(self):
        """Alignment across multiple chains should use all matched atoms."""
        s1 = Structure('s1')
        s2 = Structure('s2')
        np.random.seed(7)
        coords_a = np.random.randn(5, 3) * 10
        coords_b = np.random.randn(4, 3) * 10
        for i, (x, y, z) in enumerate(coords_a):
            s2.add_atom('A', str(i + 1), 'ALA', 'CA', 'C', x, y, z)
        for i, (x, y, z) in enumerate(coords_b):
            s2.add_atom('B', str(i + 1), 'GLY', 'CA', 'C', x, y, z)
        # Mobile = target + translation
        import copy
        s1 = copy.deepcopy(s2)
        s1.translate([42.0, -17.0, 99.0])
        result = s1.align_to(s2)
        assert result['n_matched'] == 9
        assert result['rmsd'] < 1e-10

    def test_align_all_atoms_mode(self):
        """atom_name=None should match all shared atom names."""
        s1 = Structure('s1')
        s1.add_atom('A', '1', 'ALA', 'N', 'N', 0, 0, 0)
        s1.add_atom('A', '1', 'ALA', 'CA', 'C', 1, 0, 0)
        s1.add_atom('A', '1', 'ALA', 'C', 'C', 2, 0, 0)
        s1.add_atom('A', '2', 'GLY', 'N', 'N', 3, 0, 0)
        s1.add_atom('A', '2', 'GLY', 'CA', 'C', 4, 0, 0)
        import copy
        s2 = copy.deepcopy(s1)
        s1.translate([50, 50, 50])
        result = s1.align_to(s2, atom_name=None)
        assert result['n_matched'] == 5  # all 5 atoms matched
        assert result['rmsd'] < 1e-10

    def test_align_real_cg_translated(self, cg_structure):
        """Translate real CG structure and align back — RMSD should be ~ 0."""
        import copy
        target = copy.deepcopy(cg_structure)
        cg_structure.translate([500.0, -300.0, 100.0])
        result = cg_structure.align_to(target)
        assert result['rmsd'] < 1e-8
        assert result['n_matched'] > 100

    def test_align_real_cg_rotated(self, cg_structure):
        """Rotate real CG structure and align back — RMSD should be ~ 0."""
        import copy
        target = copy.deepcopy(cg_structure)
        # Rotate 45° around Y axis
        a = np.pi / 4
        Ry = np.array([[np.cos(a), 0, np.sin(a)],
                       [0, 1, 0],
                       [-np.sin(a), 0, np.cos(a)]])
        cg_structure.rotate(Ry)
        cg_structure.translate([200, -100, 50])
        result = cg_structure.align_to(target)
        assert result['rmsd'] < 1e-6
        assert result['n_matched'] > 100


# ──────────────────────────────────────────────
# CLI / align_structures: rotation + translation + missing residues
# ──────────────────────────────────────────────

class TestAlignStructuresCLI:
    """Tests for protein_modifier.backend.align.align_structures."""

    def test_self_alignment_with_rotation_translation_and_missing_residues(self, tmp_path):
        """
        Round-trip alignment test:
            1. Take a structure (chain P of 6KN8, 274 residues).
            2. Rotate + translate it in space.
            3. Drop 20 random residues to simulate missing density.
            4. Save as a single-chain CIF and align it back to the original.
            5. Expect near-zero RMSD because a single rigid transform can
               perfectly invert the rotation+translation; missing residues
               only reduce the number of matched CAs.
        """
        from protein_modifier.backend.align import align_structures
        import copy
        import random

        # Load source crystal and pick chain P as a representative single chain.
        src = Structure.from_dict(parse_cif(ALLATOM_CIF))
        assert "P" in src.chains, "expected chain P in 6KN8 fixture"

        ref_struct = Structure(name="ref")
        ref_struct.chains["P"] = copy.deepcopy(src.chains["P"])

        # Build the mobile copy: same atoms, then rotate + translate + delete.
        mob_struct = copy.deepcopy(ref_struct)

        # Rotation: 37° about an arbitrary axis (1,2,3)/||.||
        theta = np.deg2rad(37.0)
        axis = np.array([1.0, 2.0, 3.0])
        axis /= np.linalg.norm(axis)
        K = np.array([
            [0.0,      -axis[2],  axis[1]],
            [axis[2],   0.0,     -axis[0]],
            [-axis[1],  axis[0],  0.0],
        ])
        R_apply = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        mob_struct.rotate(R_apply, center=np.zeros(3))
        mob_struct.translate([123.4, -56.7, 89.0])

        # Drop 20 random residues from the mobile chain.
        rng = random.Random(42)
        mob_chain = mob_struct.chains["P"]
        all_res_ids = list(mob_chain.residues.keys())
        assert len(all_res_ids) >= 50
        to_delete = rng.sample(all_res_ids, 20)
        for rid in to_delete:
            del mob_chain.residues[rid]
        assert len(mob_chain.residues) == len(all_res_ids) - 20

        # Write both structures to disk so we exercise the public CLI API.
        ref_path = tmp_path / "reference.cif"
        mob_path = tmp_path / "mobile.cif"
        out_path = tmp_path / "aligned.cif"
        write_cif(ref_struct.to_dict(), str(ref_path))
        write_cif(mob_struct.to_dict(), str(mob_path))

        result = align_structures(
            reference_path=str(ref_path),
            mobile_path=str(mob_path),
            output_path=str(out_path),
            ca_only=True,
        )

        # All non-deleted residues should match (sequence alignment skips gaps).
        expected_matches = len(all_res_ids) - 20
        assert result["n_matched"] == expected_matches, (
            f"expected {expected_matches} matched CAs, got {result['n_matched']}"
        )

        # A single rigid transform can perfectly invert the applied one,
        # so RMSD is bounded by the CIF coordinate write precision
        # (3 decimal places ~> ~5e-4 A noise floor).
        assert result["rmsd"] < 1e-2, (
            f"self-alignment RMSD too high: {result['rmsd']:.6f} A "
            "(rotation/translation should invert nearly exactly)"
        )

        # Output file exists and contains both chains, CA-only, equal length.
        assert out_path.exists()
        out_struct = Structure.from_dict(parse_cif(str(out_path)))
        assert set(out_struct.chains) == {"A", "B"}
        chain_a_cas = [a for r in out_struct.chains["A"] for a in r if a.name == "CA"]
        chain_b_cas = [a for r in out_struct.chains["B"] for a in r if a.name == "CA"]
        assert len(chain_a_cas) == expected_matches
        assert len(chain_b_cas) == expected_matches

        # Per-residue CA distance between chain A and chain B must be ~0.
        a_coords = np.array([[a.x, a.y, a.z] for a in chain_a_cas])
        b_coords = np.array([[a.x, a.y, a.z] for a in chain_b_cas])
        per_pair_dist = np.linalg.norm(a_coords - b_coords, axis=1)
        assert per_pair_dist.max() < 5e-2, (
            f"max per-CA distance after alignment = {per_pair_dist.max():.6f} A"
        )
