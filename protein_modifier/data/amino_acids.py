# Standard Amino Acid Mapping (3-letter to 1-letter)
AA_MAP_3_TO_1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

# Non-standard residues mapped to their standard equivalents
NONSTANDARD_AA_MAP_3_TO_1 = {
    # Selenomethionine
    'MSE': 'M',
    # Phosphorylated residues
    'SEP': 'S',  # Phosphoserine
    'TPO': 'T',  # Phosphothreonine
    'PTR': 'Y',  # Phosphotyrosine
    # Methylated residues
    'MLY': 'K',  # N-dimethyl-lysine
    'MLZ': 'K',  # N-methyl-lysine
    'M3L': 'K',  # N-trimethyl-lysine
    # Acetylated residues
    'ALY': 'K',  # N-acetyl-lysine
    # Hydroxylated residues
    'HYP': 'P',  # Hydroxyproline
    # Cysteine variants
    'CSS': 'C',  # S-mercaptocysteine
    'CSO': 'C',  # S-hydroxycysteine
    'CSD': 'C',  # S-cysteinesulfinic acid
    'CME': 'C',  # S,S-(2-hydroxyethyl)thiocysteine
    'OCS': 'C',  # Cysteinesulfonic acid
    # Histidine protonation states
    'HID': 'H',  # Delta-protonated histidine
    'HIE': 'H',  # Epsilon-protonated histidine
    'HIP': 'H',  # Doubly protonated histidine
    'HSE': 'H',  # Selenohistidine / epsilon-protonated
    'HSD': 'H',  # Delta-protonated histidine (CHARMM)
    # D-amino acids
    'DAL': 'A',  # D-alanine
    'DVA': 'V',  # D-valine
}

# Reverse mapping (1-letter to 3-letter) — standard residues only
AA_MAP_1_TO_3={
    'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE', 
    'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU', 
    'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG', 
    'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'
}

