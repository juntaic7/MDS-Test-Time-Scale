#!/usr/bin/env python3
"""
Compute CC (Consistency-Confidence) score from two LLM comparison result files.

Usage: 
    python compute_cc_score.py file1.jsonl file2.jsonl
    python compute_cc_score.py --help

Example:
    python compute_cc_score.py results/openasp/claude-3-5/llmcompare_nc_2_1.jsonl results/openasp/claude-3-5/llmcompare_nc_2_2.jsonl
"""

import argparse
import re
import sys
import os
from pathlib import Path
from utils import read_jsonl
from metrics.cc_score import compute_cc_score


def extract_decision(text):
    """Extract decision (1, 2, or tie) from LLM comparison result text."""
    # Pattern matches both bold and non-bold formats
    pattern = r'\**Decision:\**\s*(tie|1|2)'
    match = re.search(pattern, text, re.IGNORECASE)
    
    if not match:
        raise ValueError("No valid decision found in text")
    
    decision = match.group(1).lower()
    return 0 if decision == 'tie' else int(decision)


def load_comparison_results(file_path):
    """Load and parse comparison results from jsonl file."""
    print(f"Loading results from {file_path}...", file=sys.stderr)
    
    results = {}
    invalid_count = 0
    
    for result in read_jsonl(file_path):
        # Extract index from id field
        idx = int(result['id'])
        result_text = result['result']
        
        try:
            decision = extract_decision(result_text)
            results[idx] = decision
        except ValueError:
            print(f"Warning: Response {idx} is invalid - no valid decision found", file=sys.stderr)
            invalid_count += 1
            continue
    
    print(f"Loaded {len(results)} valid results, {invalid_count} invalid", file=sys.stderr)
    return results


def compute_consistency_scores(results1, results2):
    """Compute consistency metrics between two sets of comparison results.
    
    Since the files have swapped positions:
    - File1: baseline=pos1, target=pos2 
    - File2: target=pos1, baseline=pos2
    
    Consistency means: if file1 prefers option X, file2 should prefer option (3-X)
    """
    # Find common indices
    common_indices = set(results1.keys()) & set(results2.keys())
    
    if not common_indices:
        raise ValueError("No common indices found between the two result files")
    
    print(f"Computing consistency for {len(common_indices)} common samples", file=sys.stderr)
    
    # Count consistent decisions (accounting for position swapping)
    consistent_decisions = []
    target_wins = 0  # count when both consistently prefer target
    baseline_wins = 0  # count when both consistently prefer baseline
    
    for idx in common_indices:
        decision1 = results1[idx]  # 1=baseline better, 2=target better
        decision2 = results2[idx]  # 1=target better, 2=baseline better
        
        # Skip ties
        if decision1 == 0 or decision2 == 0:
            continue
            
        # Check consistency (swapped positions)
        # File1 says baseline better (1) <-> File2 says baseline better (2) = CONSISTENT
        # File1 says target better (2) <-> File2 says target better (1) = CONSISTENT
        consistent = (decision1 == 1 and decision2 == 2) or (decision1 == 2 and decision2 == 1)
        
        if consistent:
            consistent_decisions.append(idx)
            if decision1 == 2:  # Both prefer target (file1=2, file2=1)
                target_wins += 1
            else:  # Both prefer baseline (file1=1, file2=2) 
                baseline_wins += 1
    
    # Calculate consistency rate (only among non-tie decisions)
    non_tie_count = sum(1 for idx in common_indices if results1[idx] != 0 and results2[idx] != 0)
    consistency = len(consistent_decisions) / non_tie_count if non_tie_count > 0 else 0
    
    print(f"Non-tie decisions: {non_tie_count}", file=sys.stderr)
    print(f"Consistent decisions: {len(consistent_decisions)}/{non_tie_count} = {consistency:.3f}", file=sys.stderr)
    print(f"When consistent: target wins {target_wins} times, baseline wins {baseline_wins} times", file=sys.stderr)
    
    return consistency, consistent_decisions, common_indices, {"target": target_wins, "baseline": baseline_wins}


def compute_win_rates(results1, results2, common_indices):
    """Compute target win rates for both comparison sets.
    
    Target win rates:
    - File1: target wins when decision = 2 (position 2)
    - File2: target wins when decision = 1 (position 1) 
    """
    # Count total non-tie decisions for each file
    non_tie_1 = sum(1 for idx in common_indices if results1[idx] != 0)
    non_tie_2 = sum(1 for idx in common_indices if results2[idx] != 0)
    
    # Target win rates
    target_wins_file1 = sum(1 for idx in common_indices if results1[idx] == 2)
    target_wins_file2 = sum(1 for idx in common_indices if results2[idx] == 1)
    
    win_rate_1 = target_wins_file1 / non_tie_1 if non_tie_1 > 0 else 0
    win_rate_2 = target_wins_file2 / non_tie_2 if non_tie_2 > 0 else 0
    
    print(f"Target win rate file1: {target_wins_file1}/{non_tie_1} = {win_rate_1:.3f}", file=sys.stderr)
    print(f"Target win rate file2: {target_wins_file2}/{non_tie_2} = {win_rate_2:.3f}", file=sys.stderr)
    
    return win_rate_1, win_rate_2


def main():
    parser = argparse.ArgumentParser(
        description="Compute CC score from two LLM comparison result files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("file1", help="First comparison result file (.jsonl)")
    parser.add_argument("file2", help="Second comparison result file (.jsonl)")
    parser.add_argument("--output", "-o", help="Output file to save results (optional)")
    
    args = parser.parse_args()
    
    # Validate input files
    for file_path in [args.file1, args.file2]:
        if not Path(file_path).exists():
            print(f"Error: File not found: {file_path}", file=sys.stderr)
            sys.exit(1)
    
    try:
        # Load comparison results
        results1 = load_comparison_results(args.file1)
        results2 = load_comparison_results(args.file2)
        
        # Compute consistency metrics
        consistency, consistent_decisions, common_indices, preferred = compute_consistency_scores(results1, results2)
        
        # Compute win rates
        win_rate_1, win_rate_2 = compute_win_rates(results1, results2, common_indices)
        
        # Compute CC scores
        cc_high = compute_cc_score(w=max(win_rate_1, win_rate_2), c=consistency)
        cc_average = compute_cc_score(w=(win_rate_1 + win_rate_2) / 2, c=consistency)
        
        # Prepare results
        results = {
            "file1": str(args.file1),
            "file2": str(args.file2),
            "total_samples": len(common_indices),
            "consistent_decisions": len(consistent_decisions),
            "consistency": round(consistency, 4),
            "target_win_rate_file1": round(win_rate_1, 4),
            "target_win_rate_file2": round(win_rate_2, 4),
            "average_target_win_rate": round((win_rate_1 + win_rate_2) / 2, 4),
            "cc_score_high": round(cc_high, 4),
            "cc_score_average": round(cc_average, 4),
            "consistent_preferences": {
                "target_wins": preferred["target"],
                "baseline_wins": preferred["baseline"]
            }
        }
        
        # Output results
        import json
        print(json.dumps(results, indent=2))
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}", file=sys.stderr)
        
        # Summary to stderr
        print(f"\n=== CC Score Summary ===", file=sys.stderr)
        print(f"Consistency: {consistency:.3f}", file=sys.stderr)
        print(f"CC Score (high): {cc_high:.3f}", file=sys.stderr)
        print(f"CC Score (average): {cc_average:.3f}", file=sys.stderr)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()