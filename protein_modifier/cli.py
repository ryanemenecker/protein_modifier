"""Command-line interface for protein_modifier."""
import argparse
import logging
import os
import sys

from protein_modifier.modify import modify_protein
from protein_modifier.backend.sim_file_generation import write_seq_dat
from protein_modifier.backend.align import align_structures
from protein_modifier.backend.combine import combine_structures


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
    build_parser.add_argument(
        "--overwrite", action="store_true", default=False,
        help="Allow writing into an existing output directory. Default raises an error instead.",
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

    # --- combine command ---
    combine_parser = subparsers.add_parser(
        "combine",
        help=("Combine two structures into one file, automatically renaming "
              "colliding chain IDs and renumbering atoms in the output."),
    )
    combine_parser.add_argument(
        "first",
        help="First input structure (.cif or .pdb).",
    )
    combine_parser.add_argument(
        "second",
        help="Second input structure (.cif or .pdb).",
    )
    combine_parser.add_argument(
        "output",
        help="Combined output path (.cif/.mmcif or .pdb/.ent).",
    )
    combine_parser.add_argument(
        "--no-rename-chains", action="store_true", default=False,
        help=("Disable automatic chain renaming and raise an error if the "
              "two structures contain the same chain ID."),
    )

    # --- align command ---
    align_parser = subparsers.add_parser(
        "align",
        help=("Align two structures by alpha carbons and write a combined "
              "file with the reference as chain A and the mobile as chain B."),
    )
    align_parser.add_argument(
        "reference",
        help="Reference structure file (.cif or .pdb). Stays fixed; becomes chain A.",
    )
    align_parser.add_argument(
        "mobile",
        help="Mobile structure file (.cif or .pdb). Aligned onto reference; becomes chain B.",
    )
    align_parser.add_argument(
        "output",
        help="Output path (.cif/.mmcif or .pdb/.ent). Format inferred from extension.",
    )
    align_parser.add_argument(
        "--ref-chain", default=None,
        help="Restrict the reference to a single chain ID (default: use all chains).",
    )
    align_parser.add_argument(
        "--mobile-chain", default=None,
        help="Restrict the mobile structure to a single chain ID (default: use all chains).",
    )
    align_parser.add_argument(
        "--ca-only", action="store_true", default=False,
        help=("Output only the matched CA atoms used for alignment "
              "(reference -> chain A, mobile -> chain B), all post-fit. "
              "Default writes every atom of the selected chains."),
    )
    align_parser.add_argument(
        "--method", choices=("structure", "sequence"), default="structure",
        help=("How to find CA correspondences. 'structure' (default) uses "
              "pure-structural ICP on CA clouds and ignores residue "
              "identity \u2014 robust when sequences are unrelated, "
              "unknown, or numbering is unreliable. 'sequence' uses "
              "Needleman-Wunsch on the 1-letter residue sequences then "
              "Kabsch \u2014 faster and more robust to large "
              "conformational change when the inputs are the same protein "
              "or close homologs."),
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
        modify_protein(args.build_file, coarse_grain=coarse_grain, overwrite=args.overwrite)

    elif args.command == "lammps":
        if not os.path.exists(args.structure_file):
            parser.error(f"Structure file not found: {args.structure_file}")
        write_seq_dat(args.structure_file, args.output, boxdims=args.boxdims)

    elif args.command == "combine":
        if not os.path.exists(args.first):
            parser.error(f"First structure not found: {args.first}")
        if not os.path.exists(args.second):
            parser.error(f"Second structure not found: {args.second}")
        result = combine_structures(
            first_path=args.first,
            second_path=args.second,
            output_path=args.output,
            rename_chains=not args.no_rename_chains,
        )
        print(
            f"Combined {result['n_chains']} chains / {result['n_atoms']} atoms into {args.output} "
            f"(chains: {', '.join(result['chains'])})"
        )

    elif args.command == "align":
        if not os.path.exists(args.reference):
            parser.error(f"Reference structure not found: {args.reference}")
        if not os.path.exists(args.mobile):
            parser.error(f"Mobile structure not found: {args.mobile}")
        result = align_structures(
            reference_path=args.reference,
            mobile_path=args.mobile,
            output_path=args.output,
            ref_chain=args.ref_chain,
            mobile_chain=args.mobile_chain,
            ca_only=args.ca_only,
            method=args.method,
        )
        print(
            f"Aligned {result['n_matched']} CA atoms; RMSD = {result['rmsd']:.3f} A. "
            f"Wrote {args.output}"
        )


if __name__ == "__main__":
    main()
