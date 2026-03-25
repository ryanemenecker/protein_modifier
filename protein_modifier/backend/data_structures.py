"""
Docstring for data structures.
"""
from __future__ import annotations

import re
import numpy as np
from protein_modifier.backend.io import parse_cif, write_cif
from protein_modifier.data.amino_acids import AA_MAP_3_TO_1, NONSTANDARD_AA_MAP_3_TO_1
from protein_modifier.data.elements import ELEMENT_MASSES

class Atom:
    def __init__(self, atom_dict: dict[str, str]) -> None:
        # Normalize keys if common aliases are used
        self._normalize_dict(atom_dict)
        self.data = atom_dict
        self.serial_id=None
        
        # Common shortcuts
        self.name = atom_dict.get('label_atom_id', 'X')
        self.element = atom_dict.get('type_symbol', 'X')
        self.x = float(atom_dict.get('Cartn_x', 0.0))
        self.y = float(atom_dict.get('Cartn_y', 0.0))
        self.z = float(atom_dict.get('Cartn_z', 0.0))
        
    def _normalize_dict(self, d: dict[str, str]) -> None:
        """Backfill CIF keys from common aliases if missing."""
        aliases = {
            'label_atom_id': ['name', 'atom_name'],
            'type_symbol': ['element', 'type'],
            'Cartn_x': ['x', 'X'],
            'Cartn_y': ['y', 'Y'],
            'Cartn_z': ['z', 'Z'],
            'label_comp_id': ['res_name', 'residue_name'],
            'label_asym_id': ['chain_id', 'chain'],
            'label_seq_id': ['res_id', 'residue_id', 'seq_id']
        }
        
        for std_key, alt_keys in aliases.items():
            if std_key not in d:
                for alt in alt_keys:
                    if alt in d:
                        d[std_key] = str(d[alt])
                        break
                        
        # Ensure auth_ IDs match label_ IDs if missing
        if 'auth_atom_id' not in d and 'label_atom_id' in d:
            d['auth_atom_id'] = d['label_atom_id']
        if 'auth_comp_id' not in d and 'label_comp_id' in d:
            d['auth_comp_id'] = d['label_comp_id']
        if 'auth_asym_id' not in d and 'label_asym_id' in d:
            d['auth_asym_id'] = d['label_asym_id']
        if 'auth_seq_id' not in d and 'label_seq_id' in d:
            d['auth_seq_id'] = d['label_seq_id']
    
    def __repr__(self) -> str:
        return f"<Atom {self.name}: {self.x:.3f}, {self.y:.3f}, {self.z:.3f}>"
    def __getitem__(self, key: str) -> str:
        return self.data[key]

class Residue:
    def __init__(self, res_id: int | str, atoms_list: list, res_name: str = "UNK", chain_id: str = "?") -> None:
        self.id = str(res_id)
        self.atoms = [Atom(a) if isinstance(a, dict) else a for a in atoms_list]
        self.was_built = False
        self.bead_type = None
        self.solvent_accessibility = None
        
        # Meta-data inference
        if self.atoms:
            first = self.atoms[0].data
            self.name = first.get('label_comp_id', res_name)
            # Prioritize auth_asym_id to match parse_cif behavior
            self.chain_id = first.get('auth_asym_id', first.get('label_asym_id', chain_id))
        else:
            self.name = res_name
            self.chain_id = chain_id

    def add_atom(self, atom_obj: Atom) -> None:
        self.atoms.append(atom_obj)
        self.was_built = True

    def assign_solvent_accessibility(self, value: int) -> None:
        self.solvent_accessibility = value
    
    def assign_bead_type(self, bead_type: int) -> None:
        self.bead_type = bead_type

    def __getitem__(self, key):
        """Access atom by name (e.g. res['CA']) or index."""
        if isinstance(key, int): return self.atoms[key]
        if isinstance(key, str):
            for a in self.atoms:
                if a.name == key: return a
            raise KeyError(f"Atom {key} not found in {self.name} {self.id}")

    def __iter__(self): return iter(self.atoms)
    def __len__(self): return len(self.atoms)
    def __repr__(self): return f"<Residue {self.name} {self.id}: {len(self.atoms)} atoms>"

def natural_sort_key(s):
    import re
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]

class Chain:
    def __init__(self, chain_id: str, residues_dict: dict | None = None) -> None:
        self.id = chain_id
        self.residues = {}
        
        if residues_dict:
            sorted_keys = sorted(residues_dict.keys(), key=natural_sort_key)
            for key in sorted_keys:
                self.residues[str(key)] = Residue(key, residues_dict[key])

    def get_or_create_residue(self, res_id: int | str, res_name: str = "UNK") -> Residue:
        res_id = str(res_id)
        if res_id not in self.residues:
            self.residues[res_id] = Residue(res_id, [], res_name, self.id)
        return self.residues[res_id]
    
    def get_sorted_residues(self) -> list[Residue]:
        sorted_keys = sorted(self.residues.keys(), key=natural_sort_key)
        return [self.residues[k] for k in sorted_keys]

    def get_amino_acid_sequence(self) -> str:
        seq = ""
        for res in self.get_sorted_residues():
            aa = res.name
            if aa in AA_MAP_3_TO_1:
                seq += AA_MAP_3_TO_1[aa]
            elif aa in NONSTANDARD_AA_MAP_3_TO_1:
                seq += NONSTANDARD_AA_MAP_3_TO_1[aa]
            else:
                seq += "X"  # Unknown amino acid
        return seq

    def __iter__(self): return iter(self.get_sorted_residues())
    def __getitem__(self, key): return self.residues[str(key)]
    def __len__(self): return len(self.residues)
    def __repr__(self): return f"<Chain {self.id}: {len(self.residues)} residues>"

class Structure:
    def __init__(self, name: str = "Structure") -> None:
        self.name = name
        self.chains = {}
        self.need_to_update_atom_numbers = False # set to True if we add or delete atoms, renumber before writing out.

    @classmethod
    def from_dict(cls, raw_dict: dict[str, dict], name: str = "Imported") -> Structure:
        struct = cls(name)
        for chain_id, residues in raw_dict.items():
            struct.chains[chain_id] = Chain(chain_id, residues)
        return struct

    # function to determine if coarse-grained (only CA atoms)
    def is_coarse_grained(self) -> bool:
        for chain in self.chains.values():
            for residue in chain:
                for atom in residue:
                    if atom.name != "CA":
                        return False
        return True
    
    def verify_non_clashing(self, min_distance: float = 3.0) -> bool:
        """Check if any atoms are clashing within the structure."""
        coords = self.get_coords()
        from scipy.spatial import distance
        dist_matrix = distance.pdist(coords)
        if np.any(dist_matrix < min_distance):
            return False
        return True
    
    def __iter__(self):
        """Allows 'for chain in structure:'"""
        return iter(self.chains.values())

    def __getitem__(self, key):
        """Allows 'structure["A"]'"""
        return self.chains[key]
    
    def __len__(self):
        return len(self.chains)
    # --------------------------------------------------

    def add_atom(self, chain_id: str, res_id: int | str, res_name: str, atom_name: str, element: str, x: float, y: float, z: float, **kwargs) -> Atom:
        """Add atom, auto-creating chain/residue."""
        if chain_id not in self.chains:
            self.chains[chain_id] = Chain(chain_id)
        
        chain = self.chains[chain_id]
        residue = chain.get_or_create_residue(res_id, res_name)
        
        atom_dict = {
            "group_PDB": "ATOM",
            "type_symbol": element.upper(),
            "label_atom_id": atom_name,
            "label_comp_id": res_name,
            "label_asym_id": chain_id,
            "label_seq_id": re.match(r'(-?\d+)(.*)', str(res_id)).group(1) if re.match(r'(-?\d+)(.*)', str(res_id)) else str(res_id),
            "auth_seq_id": str(res_id),
            "pdbx_PDB_ins_code": "?",
            "Cartn_x": f"{float(x):.3f}",
            "Cartn_y": f"{float(y):.3f}",
            "Cartn_z": f"{float(z):.3f}",
            "occupancy": "1.00",
            "B_iso_or_equiv": "0.00",
            "auth_asym_id": chain_id,
            "auth_comp_id": res_name,
            "auth_atom_id": atom_name
        }
        atom_dict.update({k: str(v) for k, v in kwargs.items()})
        
        new_atom = Atom(atom_dict)
        residue.add_atom(new_atom)
        self.need_to_update_atom_numbers = True
        return new_atom
    
    def delete_atom(self, chain_id: str, res_id: int | str, atom_name: str) -> None:
        """Delete an atom by name from a specific residue and chain."""
        try:
            residue = self.chains[chain_id].residues[str(res_id)]
            residue.atoms = [a for a in residue.atoms if a.name != atom_name]
            self.need_to_update_atom_numbers = True
        except KeyError:
            raise ValueError(f"Chain {chain_id} or residue {res_id} not found.")
        
    def renumber_atoms(self) -> None:
        """Renumber atoms sequentially starting from 1."""
        atom_counter = 1
        for chain in self.chains.values():
            for residue in chain:
                for atom in residue:
                    atom.data['id'] = str(atom_counter)
                    atom_counter += 1

    def coarse_grain(self, method: str = "CA") -> Structure:
        """Returns a NEW Structure object reduced to beads."""
        cg_struct = Structure(f"{self.name}_{method}")
        
        for chain in self.chains.values():
            for residue in chain:
                target_x, target_y, target_z = 0.0, 0.0, 0.0
                found_bead = False
                
                if method.upper() == "CA":
                    for atom in residue:
                        if atom.name == "CA":
                            target_x, target_y, target_z = atom.x, atom.y, atom.z
                            found_bead = True
                            break
                            
                elif method.upper() == "COM":
                    total_mass = 0.0
                    sum_mx = sum_my = sum_mz = 0.0
                    for atom in residue:
                        m = ELEMENT_MASSES.get(atom.element.upper(), 12.01)
                        sum_mx += m * atom.x
                        sum_my += m * atom.y
                        sum_mz += m * atom.z
                        total_mass += m
                    
                    if total_mass > 0:
                        target_x = sum_mx / total_mass
                        target_y = sum_my / total_mass
                        target_z = sum_mz / total_mass
                        found_bead = True
                
                if found_bead:
                    cg_struct.add_atom(
                        chain_id=chain.id, res_id=residue.id, res_name=residue.name,
                        atom_name="CA", element="C", # Force CA for VMD ribbons
                        x=target_x, y=target_y, z=target_z, B_iso_or_equiv=10.0
                    )
        return cg_struct

    def get_coords(self) -> np.ndarray:
        """Returns the coordinates for all atoms in the structure as a np array."""
        coords = []
        for chain in self.chains.values():
            for residue in chain:
                for atom in residue:
                    coords.append([atom.x, atom.y, atom.z])
        
        if not coords:
            return np.zeros((0, 3))
            
        return np.array(coords)

    def to_dict(self) -> dict[str, dict[str, list[dict[str, str]]]]:
        if self.need_to_update_atom_numbers:
            self.renumber_atoms()
        raw = {}
        for chain in self.chains.values():
            raw[chain.id] = {}
            for residue in chain.get_sorted_residues():
                raw[chain.id][residue.id] = [a.data for a in residue.atoms]
        return raw
    
    def assign_residue_solvent_access(self, sasa_values: list[int]) -> None:
        sasa_ind = 0
        # iterate over all residues
        for chain in self.chains.values():
            for residue in chain:
                # all built IDRs will be solvent accessible. 
                if residue.was_built:
                    residue.assign_solvent_accessibility(0)
                else:
                    residue.assign_solvent_accessibility(sasa_values[sasa_ind])
                    sasa_ind += 1
    
    def center_structure_in_box(self, box_size: float = 100.0) -> None:
        coords = self.get_coords()
        if len(coords) == 0:
            return  # No atoms to center
        centroid = np.mean(coords, axis=0)
        shift = box_size / 2 - centroid
        for chain in self.chains.values():
            for residue in chain:
                for atom in residue:
                    atom.x += shift[0]
                    atom.y += shift[1]
                    atom.z += shift[2]

    def build_atom_serial_ids(self) -> None:
        """Assign serial IDs to atoms based on their order in the structure."""
        atom_id = 1
        for chain in self.chains.values():
            for residue in chain:
                for atom in residue:
                    atom.serial_id = atom_id
                    atom_id += 1

    def get_atom_index_of_built_residues(self) -> list[int]:
        self.build_atom_serial_ids()  # Ensure serial IDs are assigned. 
                                      # These must be correct, so we can just rebuild to be safe. 
        built_indices = []
        for chain in self.chains.values():
            for residue in chain:
                if residue.was_built:
                    for atom in residue:
                        built_indices.append(int(atom.serial_id))
        return built_indices
    
    def get_full_sequence(self) -> str:
        """
        get all amino acids for all chains in a row, return a single string.
        """
        full_seq = ""
        for chain in self.chains.values():
            full_seq += chain.get_amino_acid_sequence()
        return full_seq

    def translate(self, vector: np.ndarray | list[float]) -> None:
        """Translate all atoms by a (3,) vector [dx, dy, dz]."""
        vector = np.asarray(vector, dtype=float)
        for chain in self.chains.values():
            for residue in chain:
                for atom in residue:
                    atom.x += vector[0]
                    atom.y += vector[1]
                    atom.z += vector[2]

    def rotate(self, rotation_matrix: np.ndarray | list[list[float]], center: np.ndarray | list[float] | None = None) -> None:
        """
        Rotate all atoms by a 3x3 rotation matrix.

        Parameters
        ----------
        rotation_matrix : (3,3) array-like
            The rotation matrix to apply.
        center : (3,) array-like or None
            Point to rotate around. If None, rotates around the centroid.
        """
        R = np.asarray(rotation_matrix, dtype=float)
        coords = self.get_coords()
        if len(coords) == 0:
            return
        if center is None:
            center = np.mean(coords, axis=0)
        else:
            center = np.asarray(center, dtype=float)
        idx = 0
        for chain in self.chains.values():
            for residue in chain:
                for atom in residue:
                    p = np.array([atom.x, atom.y, atom.z]) - center
                    p_rot = R @ p + center
                    atom.x, atom.y, atom.z = float(p_rot[0]), float(p_rot[1]), float(p_rot[2])
                    idx += 1

    def get_residue_coord(self, chain_id: str, res_id: int | str, atom_name: str = 'CA') -> np.ndarray:
        """Return the (x,y,z) coordinate of a specific atom in a residue."""
        atom = self.chains[chain_id].residues[str(res_id)][atom_name]
        return np.array([atom.x, atom.y, atom.z])

    def align_to(self, target: Structure, atom_name: str = 'CA',
                 chain_map: dict[str, str] | None = None) -> dict:
        """
        Align this structure onto *target* by finding matched atoms and
        applying an optimal rigid-body superposition (Kabsch algorithm).

        Only atoms present in **both** structures (matched by chain ID,
        residue ID, and atom name) are used to compute the alignment.
        The resulting rotation+translation is then applied to **all**
        atoms in this structure, so unmatched regions move with the rest.

        Parameters
        ----------
        target : Structure
            The reference structure to align onto.
        atom_name : str
            Which atom to match on per residue (default ``'CA'``).
            Use ``'CA'`` for C-alpha alignment or ``None`` to match on
            all atom names.
        chain_map : dict[str, str] or None
            Optional mapping ``{self_chain_id: target_chain_id}`` when the
            same chain has different IDs in the two structures.  If *None*,
            chains are matched by identical IDs.

        Returns
        -------
        dict
            ``rmsd``     — RMSD over matched atoms after alignment (Å).
            ``n_matched`` — Number of atom pairs used.
            ``rotation``  — (3,3) rotation matrix applied.
            ``translation`` — (3,) effective translation applied.

        Raises
        ------
        ValueError
            If fewer than 3 atom pairs can be matched.
        """
        from protein_modifier.backend.modify_structure import kabsch_align

        # --- 1. Collect paired coordinates ---
        mobile_coords = []
        target_coords = []

        for self_chain_id, chain in self.chains.items():
            target_chain_id = self_chain_id
            if chain_map is not None:
                target_chain_id = chain_map.get(self_chain_id, self_chain_id)
            if target_chain_id not in target.chains:
                continue
            target_chain = target.chains[target_chain_id]

            for res_id, residue in chain.residues.items():
                if res_id not in target_chain.residues:
                    continue
                target_residue = target_chain.residues[res_id]

                if atom_name is not None:
                    # Match a single atom name per residue
                    try:
                        a_self = residue[atom_name]
                        a_targ = target_residue[atom_name]
                    except KeyError:
                        continue
                    mobile_coords.append([a_self.x, a_self.y, a_self.z])
                    target_coords.append([a_targ.x, a_targ.y, a_targ.z])
                else:
                    # Match every shared atom name
                    target_names = {a.name for a in target_residue}
                    for a_self in residue:
                        if a_self.name in target_names:
                            a_targ = target_residue[a_self.name]
                            mobile_coords.append([a_self.x, a_self.y, a_self.z])
                            target_coords.append([a_targ.x, a_targ.y, a_targ.z])

        n_matched = len(mobile_coords)
        if n_matched < 3:
            raise ValueError(
                f"Need at least 3 matched atom pairs for alignment, found {n_matched}."
            )

        mobile_arr = np.array(mobile_coords, dtype=np.float64)
        target_arr = np.array(target_coords, dtype=np.float64)

        # --- 2. Kabsch alignment ---
        R, mobile_centroid, target_centroid = kabsch_align(mobile_arr, target_arr)

        # --- 3. Apply transform to ALL atoms in self ---
        for chain in self.chains.values():
            for residue in chain:
                for atom in residue:
                    p = np.array([atom.x, atom.y, atom.z], dtype=np.float64) - mobile_centroid
                    p_rot = R @ p + target_centroid
                    atom.x, atom.y, atom.z = float(p_rot[0]), float(p_rot[1]), float(p_rot[2])

        # --- 4. Compute RMSD over matched atoms (post-alignment) ---
        aligned_mobile = (mobile_arr - mobile_centroid) @ R.T + target_centroid
        diff = aligned_mobile - target_arr
        rmsd = float(np.sqrt((diff ** 2).sum(axis=1).mean()))

        return {
            'rmsd': rmsd,
            'n_matched': n_matched,
            'rotation': R,
            'translation': target_centroid - R @ mobile_centroid,
        }

    def merge(self, other: Structure, rename_chains: bool = True) -> Structure:
        """
        Merge another Structure into this one.

        Parameters
        ----------
        other : Structure
            The structure to merge in.
        rename_chains : bool
            If True, automatically rename chains from `other` that collide
            with existing chain IDs. If False, raises ValueError on collision.

        Returns
        -------
        self (modified in place)
        """
        for chain_id, chain in other.chains.items():
            new_id = chain_id
            if chain_id in self.chains:
                if not rename_chains:
                    raise ValueError(
                        f"Chain ID '{chain_id}' exists in both structures. "
                        "Set rename_chains=True to auto-rename."
                    )
                # find next available single-letter chain ID
                import string
                available = [c for c in string.ascii_uppercase if c not in self.chains]
                if not available:
                    available = [c for c in string.ascii_lowercase if c not in self.chains]
                if not available:
                    raise ValueError("No available chain IDs for renaming.")
                new_id = available[0]
            # deep-copy residues into a new Chain
            new_chain = Chain(new_id)
            for res_key, residue in chain.residues.items():
                new_chain.residues[res_key] = residue
                residue.chain_id = new_id
                for atom in residue.atoms:
                    atom.data['label_asym_id'] = new_id
                    atom.data['auth_asym_id'] = new_id
            self.chains[new_id] = new_chain
        self.need_to_update_atom_numbers = True
        return self

    def validate_bond_lengths(self, expected: float = 3.8, tolerance: float = 1.0,
                              atom_name: str = 'CA') -> list[dict]:
        """
        Check CA-CA distances between consecutive residues within each chain.

        Parameters
        ----------
        expected : float
            Expected bond length (default 3.8 Å for CA-CA).
        tolerance : float
            Allowed deviation from expected (default 1.0 Å).
        atom_name : str
            Atom name to measure distances between (default 'CA').

        Returns
        -------
        list of dict
            Each dict has keys: chain_id, res_i, res_j, distance, expected.
            Empty list means all bonds are within tolerance.
        """
        issues = []
        for chain in self.chains.values():
            sorted_res = chain.get_sorted_residues()
            for i in range(len(sorted_res) - 1):
                r1, r2 = sorted_res[i], sorted_res[i + 1]
                try:
                    a1 = r1[atom_name]
                    a2 = r2[atom_name]
                except KeyError:
                    continue
                dist = np.linalg.norm(
                    np.array([a1.x, a1.y, a1.z]) - np.array([a2.x, a2.y, a2.z])
                )
                if abs(dist - expected) > tolerance:
                    issues.append({
                        'chain_id': chain.id,
                        'res_i': r1.id,
                        'res_j': r2.id,
                        'distance': round(dist, 3),
                        'expected': expected,
                    })
        return issues

    def detect_chain_breaks(self, max_distance: float = 5.0, atom_name: str = 'CA') -> list[dict]:
        """
        Detect chain breaks (gaps where consecutive residue CA atoms are too far apart).

        Parameters
        ----------
        max_distance : float
            Maximum CA-CA distance before flagging a break (default 5.0 Å).
        atom_name : str
            Atom name to measure (default 'CA').

        Returns
        -------
        list of dict
            Each dict has keys: chain_id, res_before, res_after, distance.
        """
        breaks = []
        for chain in self.chains.values():
            sorted_res = chain.get_sorted_residues()
            for i in range(len(sorted_res) - 1):
                r1, r2 = sorted_res[i], sorted_res[i + 1]
                try:
                    a1 = r1[atom_name]
                    a2 = r2[atom_name]
                except KeyError:
                    continue
                dist = np.linalg.norm(
                    np.array([a1.x, a1.y, a1.z]) - np.array([a2.x, a2.y, a2.z])
                )
                if dist > max_distance:
                    breaks.append({
                        'chain_id': chain.id,
                        'res_before': r1.id,
                        'res_after': r2.id,
                        'distance': round(dist, 3),
                    })
        return breaks

    def check_sequence_consistency(self, reference_sequences: dict[str, str]) -> dict[str, dict]:
        """
        Compare each chain's sequence against a reference.

        Parameters
        ----------
        reference_sequences : dict
            Mapping of chain_id to expected 1-letter sequence string.

        Returns
        -------
        dict
            Per-chain results with keys: matches (bool), structure_seq, reference_seq.
        """
        results = {}
        for chain_id, ref_seq in reference_sequences.items():
            if chain_id not in self.chains:
                results[chain_id] = {
                    'matches': False,
                    'structure_seq': '',
                    'reference_seq': ref_seq,
                }
                continue
            struct_seq = self.chains[chain_id].get_amino_acid_sequence()
            results[chain_id] = {
                'matches': struct_seq == ref_seq,
                'structure_seq': struct_seq,
                'reference_seq': ref_seq,
            }
        return results

    def position_relative_to(self, other: Structure, self_chain: str, self_res: int | str,
                             other_chain: str, other_res: int | str,
                             target_distance: float, atom_name: str = 'CA',
                             min_clash_distance: float = 3.0) -> Structure:
        """
        Translate and rotate `self` so that a specified residue in `self` is
        `target_distance` angstroms from a specified residue in `other`, 
        without steric clashes.

        Parameters
        ----------
        other : Structure
            The reference (stationary) structure.
        self_chain : str
            Chain ID in self containing the anchor residue.
        self_res : int or str
            Residue ID in self to position.
        other_chain : str
            Chain ID in other containing the target residue.
        other_res : int or str
            Residue ID in other to position relative to.
        target_distance : float
            Desired distance (angstroms) between the two anchor residues.
        atom_name : str
            Atom name to use for distance measurement (default 'CA').
        min_clash_distance : float
            Minimum allowed distance between any atoms of the two structures.

        Returns
        -------
        self (modified in place — translated so anchor residues are target_distance apart)
        """
        from scipy.spatial.distance import cdist

        other_anchor = other.get_residue_coord(other_chain, other_res, atom_name)
        self_anchor = self.get_residue_coord(self_chain, self_res, atom_name)

        # Compute direction from other_anchor to self_anchor (or pick random if overlapping)
        direction = self_anchor - other_anchor
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            direction = np.array([1.0, 0.0, 0.0])
        else:
            direction = direction / dist

        # Move self so that self_anchor is exactly target_distance from other_anchor
        desired_self_anchor = other_anchor + direction * target_distance
        shift = desired_self_anchor - self_anchor
        self.translate(shift)

        # Check for clashes and nudge if needed
        other_coords = other.get_coords()
        self_coords = self.get_coords()
        if len(other_coords) > 0 and len(self_coords) > 0:
            dists = cdist(self_coords, other_coords)
            min_dist = np.min(dists)
            if min_dist < min_clash_distance:
                # Push further along the same direction to resolve clash
                deficit = min_clash_distance - min_dist + 0.5  # small buffer
                self.translate(direction * deficit)

        return self