# Project Guidelines — protein_modifier

## Overview

`protein_modifier` is a Python package for modifying protein structures: building missing residues (IDRs, loops, termini) into crystal structures, combining CIF/PDB files, translating structures to avoid clashes, and generating simulation input files (LAMMPS). The main entry point is `modify_protein()` in `protein_modifier/modify.py`.

## Architecture

```
protein_modifier/
├── __init__.py               # Exports modify_protein()
├── modify.py                 # Main orchestrator — the public API
├── cli.py                    # CLI entry point (protein-modifier build/lammps)
├── backend/
│   ├── data_structures.py    # Core data model: Structure > Chain > Residue > Atom
│   ├── io.py                 # CIF/PDB parsing & writing — includes parse_structure() auto-detect
│   ├── build_idr.py          # IDR construction (N-term, C-term, loops)
│   ├── build_loop.py         # DEPRECATED — redirects to build_idr.py
│   ├── find_missing_res.py   # Missing residue detection via numbering or alignment
│   ├── modify_structure.py   # Geometric utilities (random walk, clash filtering, rotation)
│   ├── protein_math.py       # Distance calculations, sphere filtering
│   ├── parse_build_file.py   # JSON build file loader & validator
│   ├── default_parameters.py # Default bond_length, stiffness_angle, clash_distance, attempts
│   ├── sim_file_generation.py# LAMMPS data file generation (sequence.dat)
│   └── utils.py              # SASA calculation via mdtraj
├── data/
│   ├── amino_acids.py        # 3-letter ↔ 1-letter AA mappings + non-standard residues
│   ├── elements.py           # Element masses
│   ├── lammps_params.py      # Bead type indices, masses, aliphatic groups
│   └── example_json.json     # Example build file
└── tests/
    └── test_protein_modifier.py
```

### Data model hierarchy

```
Structure
  └─ Chain[chain_id]
      └─ Residue[res_id]
          └─ Atom (name, element, x, y, z, serial_id, data dict)
```

`Structure.from_dict()` creates from parsed CIF data. `Structure.to_dict()` exports back. `Structure.coarse_grain()` reduces to CA-only representation.

### Key workflow (modify_protein)

1. Parse JSON build file → validate parameters
2. Parse input CIF → build Structure
3. Coarse-grain to CA atoms
4. Detect missing residues by residue numbering against reference sequences
5. Classify gaps as N-terminal, C-terminal, or internal loop
6. Build each gap using constrained random walk with clash avoidance
7. Verify no steric clashes in final structure
8. Write output CIF + build report

## Code Style

- Python ≥ 3.8
- Max line length: 119 (flake8/yapf)
- Indent: 4 spaces, no tabs
- Formatting: yapf
- Type stubs: `py.typed` marker present

## Build and Test

```bash
# Install in development mode
pip install -e ".[test]"

# Run tests
pytest protein_modifier/tests/

# Versioning is managed by versioningit (git tags)
```

## Conventions

- **Type annotations**: All public functions have `from __future__ import annotations` and type hints.
- **Coarse-grained only for now**: `modify_protein(build_file, coarse_grain=False)` raises `NotImplementedError`. All building assumes CA-only representation.
- **Bond length**: Default 3.8 Å (CA–CA). Configurable per build file.
- **Stiffness angle**: Default 135° (90=sharp, 180=straight). Controls random walk bending.
- **Clash distance**: Default 3.4 Å. Minimum allowed distance between any two atoms.
- **Build retries**: IDR building retries up to `attempts` times (default 5, configurable in build JSON).
- **Non-standard residues**: `NONSTANDARD_AA_MAP_3_TO_1` in `amino_acids.py` maps 20+ modified residues (MSE, SEP, TPO, etc.) to their standard equivalents. Used by sequence extraction and missing residue detection.
- **CIF and PDB supported**: `parse_structure()` auto-detects format by extension. Output format follows the output path extension.
- **CIF preferred**: mmCIF is the primary format for internal use.
- **Residue numbering is 1-indexed** throughout, matching PDB convention.
- **`was_built` flag**: Residues added by the builder have `was_built=True` to distinguish from experimentally resolved residues.
- **Data dicts**: Atoms carry a raw `data` dict with CIF column keys. `Atom._normalize_dict()` handles aliased column names.

## Dependencies

Runtime: numpy, scipy (spatial.distance), mdtraj (SASA only, in utils.py)
Build: setuptools, versioningit
Test: pytest

All runtime dependencies are declared in `pyproject.toml`.

## Important Details

- `io.py` uses no external libraries — pure regex-based CIF tokenizer. `parse_structure()` dispatches to `parse_cif()` or `parse_pdb()` by extension.
- `write_cif()` sorts backbone atoms (N→CA→C→O) first for VMD ribbon rendering compatibility.
- `sim_file_generation.py` is a separate pipeline from `modify_protein()` — it generates LAMMPS `.dat` files from an already-modified structure.
- `build_loop.py` is DEPRECATED — it redirects to `build_idr.build_loop_coordinates()`. All loop building uses `build_idr.py`.
- The `data/lammps_params.py` file defines 75 bead types: 20 exposed AAs, 20 buried AAs, RNA nucleotides, aliphatic variants, and FRC.
- **Structure manipulation**: `Structure.merge()` combines two structures with chain-rename support. `Structure.translate()`, `Structure.rotate()`, and `Structure.position_relative_to()` support composing multi-protein systems.
- **Structure validation**: `Structure.validate_bond_lengths()`, `Structure.detect_chain_breaks()`, and `Structure.check_sequence_consistency()` return structured dicts for programmatic validation.
- **CLI**: `protein-modifier build <file.json>` and `protein-modifier lammps <structure> <output>` are available after installation.
- **Logging**: All modules use `logging.getLogger(__name__)`. Users configure verbosity via `logging.basicConfig()`.
- **CI/CD**: `.github/workflows/ci.yml` runs lint (flake8), tests (pytest across Python 3.9–3.12, ubuntu + macOS), and build verification.
