# protein_modifier

A Python package for modifying protein structures: building missing residues (IDRs, loops, termini) into crystal structures, combining CIF/PDB files, aligning structures in 3D, translating structures to avoid clashes, and generating LAMMPS simulation input files.

> **NOTE: PROJECT IN PROGRESS. USE AT YOUR OWN RISK.**

## Table of Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Build File Format](#build-file-format)
- [Parameter Tuning Guide](#parameter-tuning-guide)
- [Data Model](#data-model)
- [Loading & Writing Structures](#loading--writing-structures)
- [Coarse-Graining](#coarse-graining)
- [Building Missing Residues](#building-missing-residues)
- [Detecting Missing Residues](#detecting-missing-residues)
- [Structural Alignment](#structural-alignment)
- [Structure Manipulation](#structure-manipulation)
- [Structure Validation](#structure-validation)
- [Combining Structures](#combining-structures)
- [LAMMPS Simulation Files](#lammps-simulation-files)
- [Non-Standard Residues](#non-standard-residues)
- [Sequence Utilities](#sequence-utilities)
- [Logging](#logging)
- [Running Tests](#running-tests)
- [Supported Formats](#supported-formats)

## Installation

```bash
# From source (development mode)
git clone git@github.com:ryanemenecker/protein_modifier.git
cd protein_modifier
pip install -e ".[test]"
```

### Requirements

- Python ≥ 3.8
- numpy
- scipy
- mdtraj (SASA calculations only)
- tqdm

## Quickstart

### Python API

```python
from protein_modifier import modify_protein

# Build missing residues into a structure using a JSON build file
modify_protein("build_instructions.json")
```

### Command Line

```bash
# Build missing residues
protein-modifier build build_instructions.json

# Generate LAMMPS data file
protein-modifier lammps structure.cif output.dat --boxdims 800
```

## Build File Format

The build file is a JSON file that specifies the input structure, output path, chains to modify, and optional parameters.

```json
{
  "input_path": "inputs/structure.cif",
  "output_path": "outputs/modified_structure.cif",
  "chains_to_modify": [
    {
      "chain_id": "A",
      "sequence": "MTEYKLVVVGAGGVGKS..."
    },
    {
      "chain_id": "B",
      "sequence": "MDLSTTPLKKGDDDAME..."
    }
  ],
  "bond_length": 3.8,
  "stiffness_angle": 135,
  "clash_distance": 3.4,
  "attempts": 5
}
```

### Required Fields

| Field | Description |
|-------|-------------|
| `input_path` | Path to the input structure file (.cif or .pdb) |
| `output_path` | Path for the output structure file (.cif or .pdb) |
| `chains_to_modify` | List of chain objects, each with `chain_id` and `sequence` |

### Optional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bond_length` | 3.8 Å | CA–CA distance for coarse-grained building |
| `stiffness_angle` | 135° | Bond angle constraint (90=sharp turn, 180=straight) |
| `clash_distance` | 3.4 Å | Minimum allowed distance between any two atoms |
| `attempts` | 5 | Number of retry attempts for IDR building |

### Build File Validation

The build file is validated before building:
- `input_path` must point to an existing file.
- Each chain in `chains_to_modify` must have a `chain_id` and `sequence`.
- No duplicate `chain_id` values are allowed.
- Every character in `sequence` must be a valid 1-letter amino acid code.
- Numeric parameters must be within valid ranges (e.g., `stiffness_angle` between 0–180, `attempts` ≥ 1).

## Parameter Tuning Guide

- **`stiffness_angle`**: Controls how straight or bent the built IDR backbone is. Values near 180° produce extended chains; values near 90° produce compact, tightly wound chains. A value of 135° (default) gives a good balance for intrinsically disordered regions.

- **`clash_distance`**: The minimum allowed distance between any pair of atoms. Lowering this (e.g., to 3.0 Å) makes building easier for crowded structures but may produce less realistic geometry.

- **`bond_length`**: The CA–CA bond length. The standard value of 3.8 Å is appropriate for virtually all use cases.

- **`attempts`**: If building fails due to steric clashes, the builder retries up to this many times with different random walks. Increase for very crowded environments.

## Data Model

Structures are represented as a hierarchy of Python objects:

```
Structure
  └─ Chain (keyed by chain_id, e.g., "A", "B")
      └─ Residue (keyed by residue ID, e.g., "1", "42")
          └─ Atom (name, element, x, y, z, serial_id, data dict)
```

### Creating structures from files

```python
from protein_modifier.backend.data_structures import Structure
from protein_modifier.backend.io import parse_structure

struct = Structure.from_dict(parse_structure("protein.cif"))
```

### Creating structures programmatically

```python
from protein_modifier.backend.data_structures import Structure

struct = Structure("my_protein")

# add_atom auto-creates chains and residues as needed
struct.add_atom("A", "1", "ALA", "CA", "C", 0.0, 0.0, 0.0)
struct.add_atom("A", "2", "GLY", "CA", "C", 3.8, 0.0, 0.0)
struct.add_atom("B", "1", "VAL", "CA", "C", 10.0, 0.0, 0.0)
```

### Accessing structure data

```python
# Iterate over chains
for chain in struct:
    print(chain.id, len(chain.residues))

# Access by chain ID
chain_a = struct["A"]

# Get the amino acid sequence of a chain
seq = struct.chains["A"].get_amino_acid_sequence()

# Get the full concatenated sequence across all chains
full_seq = struct.get_full_sequence()

# Get all atom coordinates as an (N, 3) numpy array
coords = struct.get_coords()

# Get the coordinate of a specific residue's atom
ca_pos = struct.get_residue_coord("A", "42", atom_name="CA")

# Access individual atoms
residue = struct.chains["A"].residues["1"]
for atom in residue:
    print(atom.name, atom.x, atom.y, atom.z)

# Access atom by name within a residue
ca_atom = residue["CA"]
```

### Modifying atoms

```python
# Add an atom (auto-creates chain/residue if needed)
struct.add_atom("A", "10", "ALA", "CA", "C", 1.0, 2.0, 3.0)

# Delete an atom by name
struct.delete_atom("A", "10", "CB")
```

### Exporting structures

```python
from protein_modifier.backend.io import write_cif, write_pdb

# Export to dict (CIF-compatible nested dict)
raw_dict = struct.to_dict()

# Write to file
write_cif(raw_dict, "output.cif")
write_pdb(raw_dict, "output.pdb")
```

## Loading & Writing Structures

### Auto-detecting format

`parse_structure()` automatically detects `.cif` or `.pdb` by file extension:

```python
from protein_modifier.backend.io import parse_structure
from protein_modifier.backend.data_structures import Structure

# Works with both .cif and .pdb files
struct = Structure.from_dict(parse_structure("protein.cif"))
struct = Structure.from_dict(parse_structure("protein.pdb"))
```

### Low-level parsers

```python
from protein_modifier.backend.io import parse_cif, parse_pdb

# Parse a CIF file directly
raw_dict = parse_cif("structure.cif")

# Parse a PDB file directly
raw_dict = parse_pdb("structure.pdb")
```

Both parsers return the same nested dict format: `{chain_id: {res_id: [atom_dict, ...]}}`.

### Writing output

```python
from protein_modifier.backend.io import write_cif, write_pdb

# Write CIF (sorts backbone atoms N→CA→C→O first for VMD ribbon compatibility)
write_cif(struct.to_dict(), "output.cif")

# Write PDB (proper column alignment and TER records)
write_pdb(struct.to_dict(), "output.pdb")
```

The output format in `modify_protein()` is determined by the file extension in `output_path`.

## Coarse-Graining

Convert an all-atom structure to a coarse-grained (one-bead-per-residue) representation:

```python
struct = Structure.from_dict(parse_structure("all_atom.cif"))

# CA method: keep only C-alpha atoms (default)
cg_struct = struct.coarse_grain(method="CA")

# COM method: use center-of-mass of each residue
cg_struct = struct.coarse_grain(method="COM")

# Check if a structure is already coarse-grained
if struct.is_coarse_grained():
    print("Structure contains only CA atoms")
```

`coarse_grain()` returns a **new** `Structure` object; the original is not modified.

> **Note**: All IDR/loop building currently operates on coarse-grained (CA-only) structures. `modify_protein()` will automatically coarse-grain the input if needed.

## Building Missing Residues

### High-level API

The simplest way to build missing residues is via the JSON build file:

```python
from protein_modifier import modify_protein

modify_protein("build_instructions.json")
# Creates: output structure file + build report (.txt)
```

The builder automatically:
1. Parses and validates the build file
2. Loads the input structure and coarse-grains it
3. Detects missing residues by comparing structure residue numbering to reference sequences
4. Classifies each gap as N-terminal, C-terminal, or internal loop
5. Builds each gap using a constrained random walk with clash avoidance
6. Verifies no steric clashes in the final structure
7. Writes the output structure and a build report

### Low-level building functions

For programmatic control, use the building functions directly:

```python
from protein_modifier.backend.build_idr import (
    build_n_term_idr,
    build_c_term_idr,
    build_loop,
)

# Build an N-terminal IDR extension
struct = build_n_term_idr(
    target_structure=struct,
    chain_id="A",
    new_idr_amino_acids="MKLFFG",
    stiffness_angle=135,
    bond_length=3.8,
    clash_distance=3.4,
    attempts=5,
)

# Build a C-terminal IDR extension
struct = build_c_term_idr(
    target_structure=struct,
    chain_id="A",
    new_idr_amino_acids="GKFLMM",
    stiffness_angle=135,
    bond_length=3.8,
    clash_distance=3.4,
    attempts=5,
)

# Build an internal loop between two existing residues
struct = build_loop(
    target_structure=struct,
    chain_id="A",
    new_idr_amino_acids="MEKLF",
    ind_of_first_connecting_atom=10,
    ind_of_last_connecting_atom=16,
    stiffness_angle=135,
    bond_length=3.8,
    clash_distance=3.4,
    attempts=5,
)
```

Residues added by the builder are marked with `was_built=True` to distinguish them from experimentally resolved residues. You can retrieve their atom indices with:

```python
built_indices = struct.get_atom_index_of_built_residues()
```

## Detecting Missing Residues

Two approaches are available for identifying which residues are missing from a structure:

### By residue numbering

Compares residue numbers in the structure against the expected full-length sequence:

```python
from protein_modifier.backend.find_missing_res import get_missing_residues_by_number

missing = get_missing_residues_by_number(
    "structure.cif",
    {"A": "MTEYKLVVVGAGGVGKS..."},
    verbose=True,
)
# Returns dict with per-chain missing residue info, classified as
# N-terminal, C-terminal, or internal gaps
```

### By sequence alignment

Uses Gotoh's affine gap alignment to detect missing residues even when numbering is unreliable:

```python
from protein_modifier.backend.find_missing_res import get_missing_residues

missing = get_missing_residues(
    "structure.cif",
    {"A": "MTEYKLVVVGAGGVGKS..."},
    verbose=True,
)
```

### Sequence alignment utility

The underlying alignment function can be used standalone:

```python
from protein_modifier.backend.find_missing_res import affine_global_align

aligned_seq1, aligned_seq2 = affine_global_align(
    "MKFLGAA",
    "MFLGA",
    match=10,
    mismatch=-5,
    gap_open=-20,
    gap_extend=-1,
)
print(aligned_seq1)  # MKFLGAA
print(aligned_seq2)  # M-FLG-A
```

## Structural Alignment

Superimpose one structure onto another using the Kabsch algorithm (SVD-based rigid-body alignment). The alignment automatically handles structures with missing atoms or residues — only atoms present in **both** structures are used to compute the optimal rotation, which is then applied to all atoms.

### Align two structures

```python
from protein_modifier.backend.data_structures import Structure
from protein_modifier.backend.io import parse_structure

# Load structures
mobile = Structure.from_dict(parse_structure("model.cif"))
target = Structure.from_dict(parse_structure("reference.cif"))

# Align mobile onto target (modifies mobile in place)
result = mobile.align_to(target)

print(f"RMSD: {result['rmsd']:.3f} Å")
print(f"Matched atoms: {result['n_matched']}")
```

### Alignment options

```python
# C-alpha alignment (default) — one atom per residue
result = mobile.align_to(target, atom_name="CA")

# All-atom alignment — uses every shared atom name per residue
result = mobile.align_to(target, atom_name=None)

# Map chains with different IDs (mobile chain A → target chain X)
result = mobile.align_to(target, chain_map={"A": "X"})
```

### How atom matching works

Atoms are paired between the two structures by matching on three keys: **chain ID**, **residue ID**, and **atom name**. Only residues present in both structures contribute to the alignment. This means:

- If the mobile structure is missing residues 10–20, alignment uses all other shared residues.
- If one structure has extra chains not in the other, those chains are ignored for the fit but still get transformed.
- Use `chain_map` when the same protein chain has different chain IDs in the two files.

At least 3 matched atom pairs are required.

### Return value

`align_to()` returns a dict with:

| Key | Type | Description |
|-----|------|-------------|
| `rmsd` | float | RMSD over matched atoms after alignment (Å) |
| `n_matched` | int | Number of atom pairs used for alignment |
| `rotation` | (3,3) np.ndarray | Rotation matrix applied |
| `translation` | (3,) np.ndarray | Translation vector applied |

### Low-level Kabsch function

For aligning raw coordinate arrays without `Structure` objects:

```python
from protein_modifier.backend.modify_structure import kabsch_align
import numpy as np

# Two paired (N, 3) coordinate arrays (same length, same ordering)
mobile_coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
target_coords = mobile_coords + [10, 20, 30]  # translated copy

R, mobile_centroid, target_centroid = kabsch_align(mobile_coords, target_coords)

# Apply the transform
aligned = (mobile_coords - mobile_centroid) @ R.T + target_centroid
```

## Structure Manipulation

### Translate

Shift all atoms by a vector `[dx, dy, dz]` (in Å):

```python
struct.translate([10.0, 0.0, 0.0])
```

### Rotate

Apply a 3×3 rotation matrix. By default, rotates around the structure's centroid:

```python
import numpy as np

# Rotate 90° around Z-axis
theta = np.radians(90)
R = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta),  np.cos(theta), 0],
    [0, 0, 1]
])
struct.rotate(R)

# Rotate around a specific point
struct.rotate(R, center=[0.0, 0.0, 0.0])
```

### Center in box

Center a structure in a cubic box (useful for simulations):

```python
struct.center_structure_in_box(box_size=800.0)
```

### Position relative to another structure

Place one structure at a specified distance from another using anchor residues, with automatic clash avoidance:

```python
struct_b.position_relative_to(
    struct_a,
    self_chain="A", self_res=1,
    other_chain="A", other_res=100,
    target_distance=20.0,
)
```

## Structure Validation

### Check for steric clashes

```python
# Returns True if no atoms are closer than min_distance
is_ok = struct.verify_non_clashing(min_distance=3.4)
if not is_ok:
    print("Structure has clashing atoms!")
```

### Validate bond lengths

Check CA–CA distances between consecutive residues within each chain:

```python
issues = struct.validate_bond_lengths(expected=3.8, tolerance=1.0)
for issue in issues:
    print(f"Chain {issue['chain_id']}: residues {issue['res_i']}-{issue['res_j']} "
          f"distance={issue['distance']:.1f} Å (expected {issue['expected']})")
```

### Detect chain breaks

Find gaps where consecutive residues are too far apart:

```python
breaks = struct.detect_chain_breaks(max_distance=5.0)
for b in breaks:
    print(f"Break in chain {b['chain_id']} between {b['res_before']} and "
          f"{b['res_after']} ({b['distance']:.1f} Å)")
```

### Check sequence consistency

Verify that the structure's residue sequences match expected reference sequences:

```python
result = struct.check_sequence_consistency({
    "A": "MTEYKLVVVGAGGVGKS...",
    "B": "MDLSTTPLKKGDDDAME...",
})

for chain_id, info in result.items():
    if info["matches"]:
        print(f"Chain {chain_id}: OK")
    else:
        print(f"Chain {chain_id}: MISMATCH")
        print(f"  Structure: {info['structure_seq']}")
        print(f"  Reference: {info['reference_seq']}")
```

## Combining Structures

### Merge

Combine two structures into one. Chain ID collisions are handled automatically:

```python
from protein_modifier.backend.data_structures import Structure
from protein_modifier.backend.io import parse_structure, write_cif

struct_a = Structure.from_dict(parse_structure("protein_a.cif"))
struct_b = Structure.from_dict(parse_structure("protein_b.cif"))

# Merge struct_b into struct_a (auto-renames conflicting chain IDs)
struct_a.merge(struct_b, rename_chains=True)

# Or raise an error on chain ID collision
struct_a.merge(struct_b, rename_chains=False)  # raises ValueError on collision

write_cif(struct_a.to_dict(), "combined.cif")
```

`merge()` modifies `struct_a` in place and returns it.

## LAMMPS Simulation Files

### Generate a LAMMPS data file

```python
from protein_modifier.backend.sim_file_generation import write_seq_dat

write_seq_dat("coarse_grained.cif", "output.dat", boxdims=800)
```

### Customizing LAMMPS output

```python
write_seq_dat(
    "coarse_grained.cif",
    "output.dat",
    boxdims=800,
    num_atom_types=75,
    num_bond_types=1,
)
```

### CLI

```bash
protein-modifier lammps structure.cif output.dat --boxdims 800
```

### Bead type assignment

The LAMMPS pipeline assigns bead types based on amino acid identity and solvent exposure (buried vs. exposed), calculated from SASA (solvent-accessible surface area). If you need to assign bead types manually:

```python
from protein_modifier.backend.sim_file_generation import assign_bead_type

struct = assign_bead_type(struct, "structure.cif", probe_radius=1.4)
```

## Non-Standard Residues

The package recognizes 20+ non-standard amino acid residues and maps them to their standard equivalents. This is used automatically during sequence extraction and missing residue detection.

Supported non-standard residues include:

| Non-standard | Standard | Description |
|:-------------|:---------|:------------|
| MSE | M | Selenomethionine |
| SEP | S | Phosphoserine |
| TPO | T | Phosphothreonine |
| PTR | Y | Phosphotyrosine |
| MLY, MLZ, M3L, ALY | K | Modified lysines |
| HYP | P | Hydroxyproline |
| CSO, CSD, CME, OCS, CSS | C | Modified cysteines |
| HID, HIE, HIP, HSE, HSD | H | Histidine protonation states |
| DAL | A | D-alanine |
| DVA | V | D-valine |

Non-standard residues are mapped automatically when calling `chain.get_amino_acid_sequence()` or detecting missing residues.

To access the mapping directly:

```python
from protein_modifier.data.amino_acids import (
    AA_MAP_3_TO_1,              # Standard: {'ALA': 'A', 'CYS': 'C', ...}
    NONSTANDARD_AA_MAP_3_TO_1,  # Non-standard: {'MSE': 'M', 'SEP': 'S', ...}
    AA_MAP_1_TO_3,              # Reverse: {'A': 'ALA', 'C': 'CYS', ...}
)
```

## Sequence Utilities

### Get chain sequence

```python
seq = struct.chains["A"].get_amino_acid_sequence()
# Returns 1-letter amino acid string, e.g., "MTEYKLVVV..."
```

### Get full structure sequence

```python
full_seq = struct.get_full_sequence()
# Concatenates all chain sequences in order
```

### Sequence alignment

```python
from protein_modifier.backend.find_missing_res import affine_global_align

aln1, aln2 = affine_global_align("MKFLGAA", "MFLGA")
# aln1 = "MKFLGAA"
# aln2 = "M-FLG-A"
```

## Logging

All modules use Python's standard `logging` module. Configure verbosity:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

Set to `logging.DEBUG` for detailed build step output, or `logging.WARNING` to suppress informational messages.

## Running Tests

```bash
pip install -e ".[test]"
pytest protein_modifier/tests/
```

## Supported Formats

| Format | Read | Write | Notes |
|--------|------|-------|-------|
| mmCIF (.cif) | ✓ | ✓ | Primary format. Pure-Python parser (no external libs). |
| PDB (.pdb) | ✓ | ✓ | Full support. `parse_structure()` auto-detects by extension. |
| LAMMPS (.dat) | — | ✓ | Generated via `write_seq_dat()`. |

### Copyright

Copyright (c) 2026, Ryan Emenecker WUSTL

#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.11.
