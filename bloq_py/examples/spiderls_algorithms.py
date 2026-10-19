"""Run exact algorithms through SpiderLS, validated Bloq IR, and the T-proxy VM.

Try simon6, qaoa5, ising5, or iqp6 for larger examples; walk3 is a quick smoke run.
Unsupported layouts fail explicitly.
Requires the pinned SpiderLS checkout and spiderls.patch; see the interop guide.
"""

import argparse
import json
from pathlib import Path

from _algorithm_candidates import CANDIDATES, generate_candidate
from _interop_demo import run_demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', choices=CANDIDATES, default='walk3')
    parser.add_argument('--output-dir', type=Path, default=Path('spiderls-algorithm'))
    parser.add_argument('--shots', type=int, default=3)
    parser.add_argument('--spacing', type=int, default=3, help='upstream data-patch spacing (minimum 2)')
    parser.add_argument('--routing-margin', type=int, default=3)
    parser.add_argument('--placement-margin', type=int)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--synthesis-only', action='store_true')
    modes.add_argument('--compile-only', action='store_true')
    args = parser.parse_args()
    if args.shots < 1:
        parser.error('shots must be positive')
    if (args.spacing < 2 or args.routing_margin < 1
            or args.placement_margin is not None and args.placement_margin < 1):
        parser.error('spacing must be at least 2 and margins must be positive')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('--output-dir must be empty; each run retains its own artifacts')
    candidate = generate_candidate(args.candidate)
    result = run_demo(candidate.source_qasm, candidate.clifford_t_qasm, candidate.ideal_unitary,
                      args.output_dir, candidate.summary, compiler='spiderls', shots=args.shots,
                      synthesis_only=args.synthesis_only, compile_only=args.compile_only,
                      layout_options=dict(spacing=args.spacing, routing_margin=args.routing_margin,
                                          placement_margin=args.placement_margin))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
