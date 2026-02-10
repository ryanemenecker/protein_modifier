"""
Docstring for data structures.
"""
import numpy as np
from protein_modifier.backend.io import parse_cif, write_cif
from protein_modifier.data.amino_acids import AA_MAP_3_TO_1
from protein_modifier.data.elements import ELEMENT_MASSES

class Atom:
    def __init__(self, atom_dict):
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
        
    def _normalize_dict(self, d):
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
    
    def __repr__(self):
        return f"<Atom {self.name}: {self.x:.3f}, {self.y:.3f}, {self.z:.3f}>"
    def __getitem__(self, key):
        return self.data[key]

class Residue:
    def __init__(self, res_id, atoms_list, res_name="UNK", chain_id="?"):
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

    def add_atom(self, atom_obj):
        self.atoms.append(atom_obj)
        self.was_built = True

    def assign_solvent_accessibility(self, value):
        self.solvent_accessibility = value
    
    def assign_bead_type(self, bead_type):
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
    def __init__(self, chain_id, residues_dict=None):
        self.id = chain_id
        self.residues = {}
        
        if residues_dict:
            sorted_keys = sorted(residues_dict.keys(), key=natural_sort_key)
            for key in sorted_keys:
                self.residues[str(key)] = Residue(key, residues_dict[key])

    def get_or_create_residue(self, res_id, res_name="UNK"):
        res_id = str(res_id)
        if res_id not in self.residues:
            self.residues[res_id] = Residue(res_id, [], res_name, self.id)
        return self.residues[res_id]
    
    def get_sorted_residues(self):
        sorted_keys = sorted(self.residues.keys(), key=natural_sort_key)
        return [self.residues[k] for k in sorted_keys]

    def get_amino_acid_sequence(self):
        seq = ""
        for res in self.get_sorted_residues():
            aa = res.name
            if aa in AA_MAP_3_TO_1:
                seq += AA_MAP_3_TO_1[aa]
            else:
                seq += "X"  # Unknown amino acid
        return seq

    def __iter__(self): return iter(self.get_sorted_residues())
    def __getitem__(self, key): return self.residues[str(key)]
    def __len__(self): return len(self.residues)
    def __repr__(self): return f"<Chain {self.id}: {len(self.residues)} residues>"

class Structure:
    def __init__(self, name="Structure"):
        self.name = name
        self.chains = {}
        self.need_to_update_atom_numbers = False # set to True if we add or delete atoms, renumber before writing out.

    @classmethod
    def from_dict(cls, raw_dict, name="Imported"):
        struct = cls(name)
        for chain_id, residues in raw_dict.items():
            struct.chains[chain_id] = Chain(chain_id, residues)
        return struct

    # function to determine if coarse-grained (only CA atoms)
    def is_coarse_grained(self):
        for chain in self.chains.values():
            for residue in chain:
                for atom in residue:
                    if atom.name != "CA":
                        return False
        return True
    
    def verify_non_clashing(self, min_distance=3.0):
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

    def add_atom(self, chain_id, res_id, res_name, atom_name, element, x, y, z, **kwargs):
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
            "label_seq_id": str(res_id).replace("A", ""),
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
    
    def delete_atom(self, chain_id, res_id, atom_name):
        """Delete an atom by name from a specific residue and chain."""
        try:
            residue = self.chains[chain_id].residues[str(res_id)]
            residue.atoms = [a for a in residue.atoms if a.name != atom_name]
            self.need_to_update_atom_numbers = True
        except KeyError:
            raise ValueError(f"Chain {chain_id} or residue {res_id} not found.")
        
    def renumber_atoms(self):
        """Renumber atoms sequentially starting from 1."""
        atom_counter = 1
        for chain in self.chains.values():
            for residue in chain:
                for atom in residue:
                    atom.data['id'] = str(atom_counter)
                    atom_counter += 1

    def coarse_grain(self, method="CA"):
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

    def get_coords(self):
        """Returns the coordinates for all atoms in the structure as a np array."""
        coords = []
        for chain in self.chains.values():
            for residue in chain:
                for atom in residue:
                    coords.append([atom.x, atom.y, atom.z])
        
        if not coords:
            return np.zeros((0, 3))
            
        return np.array(coords)

    def to_dict(self):
        if self.need_to_update_atom_numbers:
            self.renumber_atoms()
        raw = {}
        for chain in self.chains.values():
            raw[chain.id] = {}
            for residue in chain.get_sorted_residues():
                raw[chain.id][residue.id] = [a.data for a in residue.atoms]
        return raw
    
    def assign_residue_solvent_access(self, sasa_values):
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
    
    def center_structure_in_box(self, box_size=100.0):
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

    def build_atom_serial_ids(self):
        """Assign serial IDs to atoms based on their order in the structure."""
        atom_id = 1
        for chain in self.chains.values():
            for residue in chain:
                for atom in residue:
                    atom.serial_id = atom_id
                    atom_id += 1

    def get_atom_index_of_built_residues(self):
        self.build_atom_serial_ids()  # Ensure serial IDs are assigned. 
                                      # These must be correct, so we can just rebuild to be safe. 
        built_indices = []
        for chain in self.chains.values():
            for residue in chain:
                if residue.was_built:
                    for atom in residue:
                        built_indices.append(int(atom.serial_id))
        return built_indices
    
    def get_full_sequence(self):
        """
        get all amino acids for all chains in a row, return a single string.
        """
        full_seq = ""
        for chain in self.chains.values():
            full_seq += chain.get_amino_acid_sequence()
        return full_seq