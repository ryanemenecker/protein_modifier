"""
Docstring for protein_modifier.backend.find_missing_res
For finding missing residues in a structure.
"""
from __future__ import annotations

import sys
import logging
from protein_modifier.backend.io import parse_cif, parse_pdb, parse_structure
from protein_modifier.backend.data_structures import Structure
from protein_modifier.data.amino_acids import AA_MAP_3_TO_1, NONSTANDARD_AA_MAP_3_TO_1

logger = logging.getLogger(__name__)


def affine_global_align(seq1: str, seq2: str, match: int = 10, mismatch: int = -5,
                        gap_open: int = -20, gap_extend: int = -1) -> tuple[str, str]:
    """
    Global Alignment with Affine Gap Penalties (Gotoh's Algorithm).
    
    Why this fixes your issue:
    - High gap_open (-20) prevents the aligner from breaking a long gap just to 
      match a single isolated residue.
    - Low gap_extend (-1) allows long missing regions (tails/loops) to exist 
      without an excessive penalty score.
    """
    n, m = len(seq1), len(seq2)
    
    # 1. Initialize Matrices
    # M = Match/Mismatch table
    # X = Gap in Seq2 (Deletions) table
    # Y = Gap in Seq1 (Insertions) table
    
    # Using float('-inf') to ensure boundaries are respected
    M = [[float('-inf')] * (m + 1) for _ in range(n + 1)]
    X = [[float('-inf')] * (m + 1) for _ in range(n + 1)]
    Y = [[float('-inf')] * (m + 1) for _ in range(n + 1)]
    
    # Traceback matrices (1=Match, 2=X, 3=Y)
    # We need separate tracebacks for each state to know which matrix we came from
    trace_M = [[0] * (m + 1) for _ in range(n + 1)]
    trace_X = [[0] * (m + 1) for _ in range(n + 1)]
    trace_Y = [[0] * (m + 1) for _ in range(n + 1)]

    # 2. Base Cases
    M[0][0] = 0
    
    # Initialize edges (Affine costs)
    for i in range(1, n + 1):
        # Cost to have a gap of length i at the start
        # Open + (i-1)*Extend
        cost = gap_open + (i - 1) * gap_extend
        M[i][0] = float('-inf') # Cannot end in Match state at boundary
        X[i][0] = cost          # Force into Delete state
        Y[i][0] = float('-inf')
        trace_X[i][0] = 1       # Came from previous extension

    for j in range(1, m + 1):
        cost = gap_open + (j - 1) * gap_extend
        M[0][j] = float('-inf')
        X[0][j] = float('-inf')
        Y[0][j] = cost          # Force into Insert state
        trace_Y[0][j] = 1

    # 3. Fill Matrices (Dynamic Programming)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score_match = (match if seq1[i-1] == seq2[j-1] else mismatch)
            
            # --- Update X (Gap in Seq2) ---
            # Can start a gap from M (open) or extend existing X (extend)
            open_x = M[i-1][j] + gap_open
            extend_x = X[i-1][j] + gap_extend
            
            if open_x >= extend_x:
                X[i][j] = open_x
                trace_X[i][j] = 0 # Came from M (New Gap)
            else:
                X[i][j] = extend_x
                trace_X[i][j] = 1 # Came from X (Extend Gap)

            # --- Update Y (Gap in Seq1) ---
            open_y = M[i][j-1] + gap_open
            extend_y = Y[i][j-1] + gap_extend
            
            if open_y >= extend_y:
                Y[i][j] = open_y
                trace_Y[i][j] = 0 # Came from M (New Gap)
            else:
                Y[i][j] = extend_y
                trace_Y[i][j] = 1 # Came from Y (Extend Gap)

            # --- Update M (Match) ---
            # Can close a gap from X or Y, or continue matching from M
            from_m = M[i-1][j-1] + score_match
            from_x = X[i-1][j-1] + score_match
            from_y = Y[i-1][j-1] + score_match
            
            best_m = max(from_m, from_x, from_y)
            M[i][j] = best_m
            
            if best_m == from_m:
                trace_M[i][j] = 0 # From M
            elif best_m == from_x:
                trace_M[i][j] = 1 # From X
            else:
                trace_M[i][j] = 2 # From Y

    # 4. Traceback
    align1, align2 = [], []
    i, j = n, m
    
    # Determine which matrix has the best score at the very end
    scores = [M[n][m], X[n][m], Y[n][m]]
    state = scores.index(max(scores)) # 0=M, 1=X, 2=Y

    while i > 0 or j > 0:
        if state == 0: # Match State (M)
            # If we are in M, we consumed a character from both
            align1.append(seq1[i-1])
            align2.append(seq2[j-1])
            
            # Where did we come from?
            next_state = trace_M[i][j]
            i -= 1
            j -= 1
            state = next_state
            
        elif state == 1: # Gap in Seq2 (X) - Deletion
            # We consumed seq1, gap in seq2
            align1.append(seq1[i-1])
            align2.append('-')
            
            # Did we just open this gap or extend it?
            from_extend = trace_X[i][j] # 0=New(from M), 1=Extend(from X)
            i -= 1
            state = 1 if from_extend == 1 else 0
            
        elif state == 2: # Gap in Seq1 (Y) - Insertion
            # We consumed seq2, gap in seq1
            align1.append('-')
            align2.append(seq2[j-1])
            
            # Did we just open this gap or extend it?
            from_extend = trace_Y[i][j] # 0=New(from M), 1=Extend(from Y)
            j -= 1
            state = 2 if from_extend == 1 else 0

    return "".join(reversed(align1)), "".join(reversed(align2))

def _build_blocks_from_states(residue_states):
    blocks = {}
    if not residue_states:
        return blocks
        
    block_id = 0
    current_status = residue_states[0]['status']
    current_start = residue_states[0]['ref_idx']
    current_seq = [residue_states[0]['char']]

    for i in range(1, len(residue_states)):
        state = residue_states[i]
        status = state['status']
        idx = state['ref_idx']
        char = state['char']

        if status == current_status:
            current_seq.append(char)
        else:
            # Close block
            blocks[block_id] = {
                'status': current_status,
                'index': [current_start, idx],
                'sequence': "".join(current_seq)
            }
            block_id += 1
            
            # Start new block
            current_status = status
            current_start = idx
            current_seq = [char]
            
    # Close final block
    blocks[block_id] = {
        'status': current_status,
        'index': [current_start, current_start + len(current_seq)],
        'sequence': "".join(current_seq)
    }
    return blocks

def _print_report_for_blocks(chain_id, blocks):
    missing_count = 0
    missing_ranges = []
    
    for b in blocks.values():
        if b['status'] == 'missing':
            count = len(b['sequence'])
            missing_count += count
            start = b['index'][0] + 1
            end = b['index'][1]
            if count == 1:
                missing_ranges.append(f"{start}")
            else:
                missing_ranges.append(f"{start}-{end}")

    if missing_count > 0:
        logger.info(f"Chain {chain_id}: {missing_count} residues missing.")
        logger.info(f"  Missing: {', '.join(missing_ranges)}")
    else:
        logger.info(f"Chain {chain_id}: Complete.")

def get_missing_residues_by_number(structure_path: str, reference_sequences: dict[str, str],
                                   verbose: bool = False) -> dict:
    """
    Identifies missing residues by verifying residue numbers (indexes) in the structure
    match the expected sequence.
    
    Args:
        structure_path: Path to .cif or .pdb file
        reference_sequences: Dict {chain_id: sequence_string}
        verbose: If True, prints warnings about sequence mismatches.
    """
    try:
        raw_dict = parse_structure(structure_path)
        structure = Structure.from_dict(raw_dict)
    except NameError:
        logger.error("parse_structure or io class not defined.")
        return {}
    except Exception as e:
        logger.error(f"Error loading structure: {e}")
        return {}

    if verbose:
        logger.info(f"--- Analyzing Missing Residues (By Number) for {structure_path} ---")
    results = {}

    for chain in structure:
        chain_id = chain.id
        
        if chain_id not in reference_sequences:
            if verbose: logger.info(f"Chain {chain_id}: No reference sequence provided, skipping.")
            continue
            
        ref_seq = reference_sequences[chain_id]
        residue_states = []

        # Iterate 1-based index corresponding to sequence
        for i, ref_char in enumerate(ref_seq, start=1):
            res_id_str = str(i)
            status = 'missing'
            
            # Check if residue ID exists in the chain
            if res_id_str in chain.residues:
                res = chain.residues[res_id_str]
                
                # Validation: Check if residue type matches expectations
                res_name_3 = res.name
                res_char_struct = AA_MAP_3_TO_1.get(res_name_3, NONSTANDARD_AA_MAP_3_TO_1.get(res_name_3, '?'))
                
                if res_char_struct != ref_char:
                    if verbose:
                        logger.warning(f"Chain {chain_id} Res {i}: Structure has {res_name_3}({res_char_struct}), Expected {ref_char}")
                
                status = 'present'

            residue_states.append({
                'ref_idx': i - 1, # 0-based index
                'char': ref_char,
                'status': status
            })

        blocks = _build_blocks_from_states(residue_states)
        results[chain_id] = blocks
        
        # Report
        if verbose:
            _print_report_for_blocks(chain_id, blocks)

    return results

def get_missing_residues(structure_path: str, reference_sequences: dict[str, str],
                         verbose: bool = False) -> dict:
    """
    Identifies missing residues using custom parser and custom alignment.
    
    Args:
        structure_path: Path to .cif or .pdb file
        reference_sequences: Dict {chain_id: sequence_string}
        verbose: If True, prints detailed alignment and block info.
    """
    
    # 1. Load Data using your custom tools
    # Ensure parse_cif_atoms and Structure are available in scope
    try:
        raw_dict = parse_structure(structure_path)
        structure = Structure.from_dict(raw_dict)
    except NameError:
        logger.error("parse_structure or io class not defined.")
        return {}
    except Exception as e:
        logger.error(f"Error loading structure: {e}")
        return {}
    
    if verbose:
        logger.info(f"--- Analyzing Missing Residues for {structure_path} ---")
    results = {}

    # 2. Iterate Chains
    for chain in structure:
        chain_id = chain.id
        
        if chain_id not in reference_sequences:
            if verbose:                
                logger.info(f"Chain {chain_id}: No reference sequence provided, skipping.")
            continue
            
        ref_seq = reference_sequences[chain_id]
        
        # 3. Extract Resolved Sequence
        res_seq_list = []
        for res in chain:
            if res.name in AA_MAP_3_TO_1:
                res_seq_list.append(AA_MAP_3_TO_1[res.name])
            elif res.name in NONSTANDARD_AA_MAP_3_TO_1:
                res_seq_list.append(NONSTANDARD_AA_MAP_3_TO_1[res.name])
            # We ignore HETATM/Water silently
        
        resolved_seq = "".join(res_seq_list)
        
        if not resolved_seq:
            if verbose:
                logger.info(f"Chain {chain_id}: No standard residues found, skipping.")
            continue

        # 4. Perform Alignment (Pure Python)
        # We tune penalties to prefer finding big gaps rather than scattering single gaps.
        # Match=10, Mismatch=-5, Gap=-5 encourages alignment of blocks.
        aligned_ref, aligned_struct = affine_global_align(
            ref_seq, resolved_seq, match=10, mismatch=-5, gap_open=-20, gap_extend=-1
        )
        
        # 5. Analyze Gaps into Blocks
        residue_states = []
        ref_idx = 0

        for r_char, s_char in zip(aligned_ref, aligned_struct):
            if r_char != '-':
                status = 'missing' if s_char == '-' else 'present'
                residue_states.append({
                    'ref_idx': ref_idx,
                    'char': r_char,
                    'status': status
                })
                ref_idx += 1

        blocks = _build_blocks_from_states(residue_states)
        results[chain_id] = blocks

        # 6. Print Report
        if verbose:
             _print_report_for_blocks(chain_id, blocks)

    return results

def group_residues_to_dict(data_list: list[tuple[int, str]]) -> dict[str, str]:
    """(Helper) Converts [(1,'M'),(2,'A')...] to {'1-2': 'MA'}"""
    if not data_list: return {}
    ranges = {}
    
    if not data_list: return {}
    
    start_idx, start_char = data_list[0]
    prev_idx = start_idx
    curr_seq = [start_char]
    
    for idx, char in data_list[1:]:
        if idx == prev_idx + 1:
            curr_seq.append(char)
        else:
            key = f"{start_idx}-{prev_idx}" if start_idx != prev_idx else f"{start_idx}"
            ranges[key] = "".join(curr_seq)
            start_idx = idx
            curr_seq = [char]
        prev_idx = idx
        
    key = f"{start_idx}-{prev_idx}" if start_idx != prev_idx else f"{start_idx}"
    ranges[key] = "".join(curr_seq)
    
    return ranges

