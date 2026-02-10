"""
Docstring for io
NOTE: PDB and CIF parsers use the same dictionary format so everything
should in theory be interchangable to allow for max flexibility. 
The write functions are separate to allow for format-specific formatting 
(e.g. strict column widths for PDB, quoting for CIF).
"""
import os
import re
from collections import defaultdict

def parse_cif(file_path):
    """
    Parses a .cif file and returns a dictionary structured by Chain -> Residue -> Atoms.
    
    Structure:
    {
        "ChainID": {
            "ResidueID": [
                { "atom_name": "N", "x": 12.34, ... (all fields) },
                ...
            ]
        }
    }
    """
    
    # 1. Regex to split CIF lines correctly respecting quotes
    # Matches: "double quoted", 'single quoted', or non-whitespace
    token_pattern = re.compile(r'(?:"([^"]*)"|\'([^\']*)\'|(\S+))')

    def tokenize(line):
        """Splits a CIF line into tokens, stripping quotes."""
        return [m[0] or m[1] or m[2] for m in token_pattern.findall(line)]

    atom_headers = []
    parsing_atoms = False
    
    # The main data structure
    # Defaultdict allows easy auto-creation of nested dicts/lists
    structure_data = defaultdict(lambda: defaultdict(list))

    with open(file_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines or comments
        if not line or line.startswith('#'):
            i += 1
            continue

        # Detect the start of a loop
        if line == 'loop_':
            # check the next lines to see if this is the _atom_site loop
            temp_headers = []
            j = i + 1
            is_atom_loop = False
            
            while j < len(lines):
                check_line = lines[j].strip()
                if check_line.startswith('_atom_site.'):
                    is_atom_loop = True
                    # Remove the prefix '_atom_site.' to make keys cleaner
                    header_name = check_line.replace('_atom_site.', '')
                    temp_headers.append(header_name)
                    j += 1
                elif check_line.startswith('_'):
                    # Different loop category
                    break
                else:
                    # End of headers, start of data
                    break
            
            if is_atom_loop:
                atom_headers = temp_headers
                parsing_atoms = True
                i = j # Move main index to data section
                continue
            else:
                i = j
                continue

        # Parse Atom Data Rows
        if parsing_atoms:
            # If we hit a new loop, hash, or global key, stop parsing atoms
            if line.startswith('loop_') or line.startswith('#') or line.startswith('_'):
                parsing_atoms = False
                continue

            tokens = tokenize(line)
            
            # Ensure line matches header count (basic validation)
            if len(tokens) != len(atom_headers):
                i += 1
                continue

            # Create a dictionary for this atom mapping {header: value}
            atom_dict = {k: v for k, v in zip(atom_headers, tokens)}

            # --- EXTRACT KEYS FOR HIERARCHY ---
            
            # 1. Get Chain ID (Prioritize auth_asym_id, fallback to label_asym_id)
            chain_id = atom_dict.get('auth_asym_id', atom_dict.get('label_asym_id', 'UNKNOWN'))

            # 2. Get Residue Unique ID
            # We construct a unique key: [SeqNum] + [InsertionCode]
            # CIF uses '.' or '?' for missing data.
            seq_id = atom_dict.get('auth_seq_id', atom_dict.get('label_seq_id', '0'))
            
            # Handle Insertion Codes (pdbx_PDB_ins_code)
            ins_code = atom_dict.get('pdbx_PDB_ins_code', '?')
            if ins_code in ['.', '?']:
                ins_code = ""
            
            # Final Residue Key (e.g., "101" or "101A")
            residue_key = f"{seq_id}{ins_code}"

            # 3. Store Data
            structure_data[chain_id][residue_key].append(atom_dict)

        i += 1

    # Convert defaultdict back to standard dict for cleaner printing/usage
    return {k: dict(v) for k, v in structure_data.items()}


def write_cif(structure_dict, output_path, data_block_name="STRUCTURE"):
    """
    Writes a strict, VMD-compatible mmCIF file.
    
    Features:
    - Sorts atoms by Chain -> Residue -> Atom Order (N, CA, C, O) for valid bond calculation.
    - Aligns columns perfectly for parser stability.
    - Includes both 'label' and 'auth' identifiers to satisfy strict parsers.
    """
    
    # 1. Define the Strict Schema (Order matches standard PDBx/mmCIF files)
    # VMD looks for specific columns to determine how to render chains.
    cif_columns = [
        "group_PDB",          # ATOM
        "id",                 # Serial ID
        "type_symbol",        # Element (C, N, O)
        "label_atom_id",      # Atom Name
        "label_alt_id",       # Alt Loc
        "label_comp_id",      # Residue Name
        "label_asym_id",      # Chain ID (System)
        "label_entity_id",    # Entity ID (default 1)
        "label_seq_id",       # Residue Seq Num (System)
        "pdbx_PDB_ins_code",  # Insertion Code
        "Cartn_x",            # Coordinates
        "Cartn_y", 
        "Cartn_z", 
        "occupancy", 
        "B_iso_or_equiv", 
        "auth_seq_id",        # Residue Seq Num (Author/PDB)
        "auth_comp_id",       # Residue Name (Author)
        "auth_asym_id",       # Chain ID (Author)
        "auth_atom_id"        # Atom Name (Author)
    ]

    # Standard Backbone Order for sorting (Critical for VMD Ribbons)
    atom_order_priority = {"N": 0, "CA": 1, "C": 2, "O": 3}

    # 2. Flatten and Sort Data
    rows = []
    serial_id = 1
    
    # Sort Chains
    for chain_id in sorted(structure_dict.keys()):
        residues = structure_dict[chain_id]
        
        # Sort Residues (Handle "10" vs "10A")
        def residue_sort_key(res_key):
            import re
            # Split "10A" into (10, "A")
            match = re.match(r"(-?\d+)(.*)", str(res_key))
            if match:
                return (int(match.group(1)), match.group(2))
            return (0, res_key)

        sorted_res_keys = sorted(residues.keys(), key=residue_sort_key)

        for res_key in sorted_res_keys:
            atom_list = residues[res_key]
            
            # Sort Atoms: Backbone first, then alphabetical
            # This helps VMD connect the ribbon correctly
            def atom_sort_key(atom):
                name = atom.get('label_atom_id', 'X')
                return (atom_order_priority.get(name, 99), name)
            
            atom_list = sorted(atom_list, key=atom_sort_key)

            for atom in atom_list:
                # 3. Prepare Row Data
                row = {}
                
                # Extract clean values or defaults
                # We use the existing dict but sanitize it for the Schema
                
                # IDs
                row["group_PDB"] = atom.get("group_PDB", "ATOM")
                row["id"] = str(serial_id)
                
                # Atom Name
                name = atom.get("label_atom_id", atom.get("auth_atom_id", "X"))
                row["label_atom_id"] = name
                row["auth_atom_id"] = name
                
                # Element (type_symbol) - CRITICAL
                # If missing, derive from atom name (e.g., CA -> C)
                if "type_symbol" in atom:
                    row["type_symbol"] = atom["type_symbol"]
                else:
                    # Strip leading digits (1H -> H) and trailing chars
                    elem = ''.join([c for c in name if c.isalpha()])
                    row["type_symbol"] = elem[0] if elem else "X"

                # Residue Name
                res_name = atom.get("label_comp_id", atom.get("auth_comp_id", "UNK"))
                row["label_comp_id"] = res_name
                row["auth_comp_id"] = res_name
                
                # Chain ID
                # Note: label_asym_id is usually strict (A, B, C), auth can be anything.
                # We normalize them to ensure VMD groups them.
                row["label_asym_id"] = chain_id
                row["auth_asym_id"] = chain_id
                
                row["label_entity_id"] = "1"

                # Residue Number
                # Extract numeric part for label_seq_id (must be int in standard CIF)
                import re
                num_match = re.match(r"(-?\d+)", str(res_key))
                seq_num = num_match.group(1) if num_match else "0"
                row["label_seq_id"] = seq_num
                row["auth_seq_id"] = seq_num # Auth can be same as label for export simplicity

                # Insertion Code
                # CIF Standard: '?' if missing, not '.'
                ins_code = atom.get("pdbx_PDB_ins_code", "?")
                if ins_code in [".", "", " "]: ins_code = "?"
                row["pdbx_PDB_ins_code"] = ins_code

                # Alt Loc
                alt = atom.get("label_alt_id", ".")
                row["label_alt_id"] = "." if alt == "" else alt

                # Coordinates (Strict Formatting)
                try:
                    row["Cartn_x"] = f"{float(atom.get('Cartn_x', 0)):.3f}"
                    row["Cartn_y"] = f"{float(atom.get('Cartn_y', 0)):.3f}"
                    row["Cartn_z"] = f"{float(atom.get('Cartn_z', 0)):.3f}"
                    row["occupancy"] = f"{float(atom.get('occupancy', 1.0)):.2f}"
                    row["B_iso_or_equiv"] = f"{float(atom.get('B_iso_or_equiv', 0.0)):.2f}"
                except ValueError:
                    row["Cartn_x"] = "0.000"
                    row["Cartn_y"] = "0.000"
                    row["Cartn_z"] = "0.000"
                    row["occupancy"] = "1.00"
                    row["B_iso_or_equiv"] = "0.00"

                rows.append(row)
                serial_id += 1

    if not rows:
        print("Error: No atoms to write.")
        return

    # 4. Calculate Column Widths for Alignment
    # Initialize with header lengths
    col_widths = {k: len(k) for k in cif_columns}
    
    # Update with max data lengths
    for r in rows:
        for k in cif_columns:
            val_len = len(r[k])
            # Check for necessary quoting which adds 2 chars
            if " " in r[k]: val_len += 2
            if val_len > col_widths[k]:
                col_widths[k] = val_len

    # 5. Write File
    with open(output_path, 'w') as f:
        f.write(f"data_{data_block_name}\n")
        f.write("#\n")
        f.write("loop_\n")
        
        # Write Headers
        for k in cif_columns:
            f.write(f"_atom_site.{k}\n")
            
        # Write Data Rows
        for r in rows:
            line_parts = []
            for k in cif_columns:
                val = r[k]
                width = col_widths[k]
                
                # CIF Quoting Logic
                if " " in val:
                    if '"' in val:
                        val = f"'{val}'"
                    else:
                        val = f'"{val}"'
                
                # Left align value in the column width + 2 spaces padding
                line_parts.append(f"{val:<{width+2}}")
            
            f.write("".join(line_parts).rstrip() + "\n")
        
        # VMD requires a clean close of the loop sometimes
        f.write("#\n")

    print(f"Successfully exported {len(rows)} atoms to {output_path}")

def write_pdb(structure_dict, output_path):
    """
    Writes a strict, VMD-compatible PDB file.
    
    Features:
    - Sorts atoms by Chain -> Residue -> Atom Order (N, CA, C, O) for valid ribbons.
    - formats columns to strict PDB width specifications (80 chars).
    - Handles proper atom name alignment (columns 13-16).
    - Adds TER records at the end of chains.
    """

    # Standard Backbone Order for sorting (Critical for VMD Ribbons)
    atom_order_priority = {"N": 0, "CA": 1, "C": 2, "O": 3}

    rows = []
    serial_id = 1
    
    # 1. Flatten and Sort Data (Identical logic to write_cif)
    sorted_chain_ids = sorted(structure_dict.keys())

    with open(output_path, 'w') as f:
        f.write("REMARK   1 CREATED BY PYTHON SCRIPT\n")

        for chain_id in sorted_chain_ids:
            residues = structure_dict[chain_id]
            
            # Helper: Sort Residues (Handle "10" vs "10A")
            def residue_sort_key(res_key):
                import re
                match = re.match(r"(-?\d+)(.*)", str(res_key))
                if match:
                    return (int(match.group(1)), match.group(2))
                return (0, res_key)

            sorted_res_keys = sorted(residues.keys(), key=residue_sort_key)

            for res_key in sorted_res_keys:
                atom_list = residues[res_key]
                
                # Helper: Sort Atoms (Backbone first)
                def atom_sort_key(atom):
                    # Prefer label, fallback to auth, fallback to X
                    name = atom.get('label_atom_id', atom.get('auth_atom_id', 'X'))
                    return (atom_order_priority.get(name, 99), name)
                
                atom_list = sorted(atom_list, key=atom_sort_key)

                for atom in atom_list:
                    # 2. Extract Data
                    
                    # Record Type (ATOM or HETATM)
                    record_type = atom.get("group_PDB", "ATOM")
                    if len(record_type) > 6: record_type = "ATOM" # Safety
                    
                    # Atom Name
                    name = atom.get("label_atom_id", atom.get("auth_atom_id", "X"))
                    
                    # Alt Loc
                    alt_loc = atom.get("label_alt_id", " ")
                    if alt_loc in [".", "?"]: alt_loc = " "
                    
                    # Residue Name
                    res_name = atom.get("label_comp_id", atom.get("auth_comp_id", "UNK"))
                    
                    # Chain ID (Truncate to 1 char for PDB)
                    chain_char = chain_id[0] if chain_id else "A"
                    
                    # Residue Seq Num
                    import re
                    num_match = re.match(r"(-?\d+)", str(res_key))
                    res_seq = int(num_match.group(1)) if num_match else 0
                    
                    # Insertion Code
                    ins_code = atom.get("pdbx_PDB_ins_code", " ")
                    if ins_code in [".", "?", ""]: ins_code = " "

                    # Coordinates
                    try:
                        x = float(atom.get('Cartn_x', 0))
                        y = float(atom.get('Cartn_y', 0))
                        z = float(atom.get('Cartn_z', 0))
                        occ = float(atom.get('occupancy', 1.0))
                        tfac = float(atom.get('B_iso_or_equiv', 0.0))
                    except ValueError:
                        x, y, z, occ, tfac = 0.0, 0.0, 0.0, 1.0, 0.0

                    # Element Symbol
                    if "type_symbol" in atom:
                        element = atom["type_symbol"]
                    else:
                        # Strip digits/chars to guess element
                        elem_clean = ''.join([c for c in name if c.isalpha()])
                        element = elem_clean[0:2] if elem_clean else "X"
                    element = element.upper()

                    # 3. Format PDB Line (Fixed Width)
                    
                    # Atom Name Alignment Logic:
                    # - 4 chars: columns 13-16 (e.g. "HG11")
                    # - <4 chars: columns 14-16 (e.g. " CA ")
                    if len(name) >= 4:
                        fmt_name = f"{name[:4]}"
                    else:
                        fmt_name = f" {name:<3}" # Space prepended, left align remaining

                    # Serial ID wrapping (PDB format breaks > 99999)
                    # We modulo 100000 to keep alignment safe
                    fmt_serial = serial_id % 100000

                    # PDB Line Format String
                    # Cols 1-6   : Record name
                    # Cols 7-11  : Serial
                    # ...        : Space
                    # Cols 13-16 : Atom Name (Handled above)
                    # Col  17    : AltLoc
                    # Cols 18-20 : ResName
                    # ...        : Space
                    # Col  22    : ChainID
                    # Cols 23-26 : ResSeq
                    # Col  27    : InsCode
                    # ...        : Spaces
                    # Cols 31-38 : X
                    # Cols 39-46 : Y
                    # Cols 47-54 : Z
                    # Cols 55-60 : Occ
                    # Cols 61-66 : B-factor
                    # ...        : Spaces
                    # Cols 77-78 : Element
                    
                    line = (
                        f"{record_type:<6}"
                        f"{fmt_serial:>5} "
                        f"{fmt_name}"
                        f"{alt_loc:1}"
                        f"{res_name:>3} "
                        f"{chain_char:1}"
                        f"{res_seq:>4}"
                        f"{ins_code:1}   "
                        f"{x:8.3f}"
                        f"{y:8.3f}"
                        f"{z:8.3f}"
                        f"{occ:6.2f}"
                        f"{tfac:6.2f}          "
                        f"{element:>2}"
                    )
                    
                    f.write(line + "\n")
                    serial_id += 1
            
            # 4. Write TER record (End of Chain)
            # VMD uses this to disconnect the ribbon between chains
            f.write(f"TER   {serial_id:>5}      {res_name:>3} {chain_char:1}{res_seq:>4}\n")
            serial_id += 1

        f.write("END\n")

    print(f"Successfully exported {serial_id - 1} atoms to {output_path}")

import os
from collections import defaultdict

def parse_pdb(file_path):
    """
    Parses a .pdb file and returns a dictionary structured by Chain -> Residue -> Atoms.
    
    Structure (Matches parse_cif output keys for interoperability):
    {
        "ChainID": {
            "ResidueKey": [
                { "label_atom_id": "CA", "Cartn_x": "12.34", ... },
                ...
            ]
        }
    }
    """
    
    # The main data structure
    structure_data = defaultdict(lambda: defaultdict(list))

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDB file not found: {file_path}")

    with open(file_path, 'r') as f:
        for line in f:
            # PDB files strictly use the first 6 characters for the Record Name
            record_type = line[0:6].strip()

            # We only care about atomic coordinates
            if record_type in ("ATOM", "HETATM"):
                
                # --- 1. SLICE DATA (Strict PDB Column format) ---
                # PDB is fixed-width, not whitespace-delimited.
                # Slicing is 0-indexed (Start, End)
                
                # Atom Serial (7-11)
                serial_id = line[6:11].strip()
                
                # Atom Name (13-16)
                # Note: PDB atom names often have leading spaces for alignment.
                atom_name = line[12:16].strip()
                
                # Alt Loc (17)
                alt_loc = line[16].strip()
                
                # Residue Name (18-20)
                res_name = line[17:20].strip()
                
                # Chain ID (22)
                chain_id = line[21].strip()
                if not chain_id: chain_id = "A" # Default if missing
                
                # Residue Seq Num (23-26)
                res_seq = line[22:26].strip()
                
                # Insertion Code (27)
                ins_code = line[26].strip()
                
                # Coordinates
                x = line[30:38].strip()
                y = line[38:46].strip()
                z = line[46:54].strip()
                
                # Occupancy & B-factor
                occ = line[54:60].strip()
                tfac = line[60:66].strip()
                
                # Element (77-78)
                element = line[76:78].strip()

                # --- 2. MAP TO COMMON SCHEMA ---
                # We use the same keys as parse_cif so downstream scripts work for both.
                atom_dict = {
                    "group_PDB": record_type,
                    "id": serial_id,
                    
                    # Identifiers
                    "label_atom_id": atom_name,
                    "auth_atom_id": atom_name,
                    "label_comp_id": res_name,
                    "auth_comp_id": res_name,
                    "label_asym_id": chain_id,
                    "auth_asym_id": chain_id,
                    "label_seq_id": res_seq,
                    "auth_seq_id": res_seq,
                    
                    # Metadata
                    "label_alt_id": alt_loc if alt_loc else ".",
                    "pdbx_PDB_ins_code": ins_code if ins_code else "?",
                    "type_symbol": element,
                    
                    # Coordinates (Stored as strings to match parse_cif behavior)
                    "Cartn_x": x,
                    "Cartn_y": y,
                    "Cartn_z": z,
                    "occupancy": occ if occ else "1.00",
                    "B_iso_or_equiv": tfac if tfac else "0.00"
                }

                # --- 3. HIERARCHY CONSTRUCTION ---
                
                # Construct Unique Residue Key: [SeqNum] + [InsertionCode]
                # Matches the logic in parse_cif (e.g., "10" or "10A")
                residue_key = f"{res_seq}{ins_code}"

                structure_data[chain_id][residue_key].append(atom_dict)

    # Convert defaultdict back to standard dict
    return {k: dict(v) for k, v in structure_data.items()}

