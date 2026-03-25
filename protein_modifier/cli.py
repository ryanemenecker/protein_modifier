"""Command-line interface for protein_modifier."""
import argparse
import logging
import os
import sys

from protein_modifier.modify import modify_protein
from protein_modifier.backend.sim_file_generation import write_seq_dat


def main():
    parser = argparse.ArgumentParser(
        prog="protein-modifier",
        description="Modify protein structures: build missing IDRs, generate LAMMPS files.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose (DEBUG) logging output.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- build command ---
    build_parser = subparsers.add_parser(
        "build",
        help="Build missing residues into a structure using a JSON build file.",
    )
    build_parser.add_argument(
        "build_file",
        help="Path to the JSON build file.",
    )
    build_parser.add_argument(
        "--fine-grain", action="store_true", default=False,
        help="Use fine-grained (all-atom) building. (Not yet implemented.)",
    )

    # --- lammps command ---
    lammps_parser = subparsers.add_parser(
        "lammps",
        help="Generate a LAMMPS .dat file from a structure.",
    )
    lammps_parser.add_argument(
        "structure_file",
        help="Path to the input structure file (.cif or .pdb).",
    )
    lammps_parser.add_argument(
        "output",
        help="Path for the output LAMMPS .dat file.",
    )
    lammps_parser.add_argument(
        "--boxdims", type=float, default=800,
        help="Simulation box dimension in angstroms (default: 800).",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
    )

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "build":
        if not os.path.exists(args.build_file):
            parser.error(f"Build file not found: {args.build_file}")
        coarse_grain = not args.fine_grain
        modify_protein(args.build_file, coarse_grain=coarse_grain)

    elif args.command == "lammps":
        if not os.path.exists(args.structure_file):
            parser.error(f"Structure file not found: {args.structure_file}")
        write_seq_dat(args.structure_file, args.output, boxdims=args.boxdims)


if __name__ == "__main__":
    main()
