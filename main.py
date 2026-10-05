"""Command-line entry point; no import-time simulation side effects."""
import argparse
from verified_simulator import VerifiedCategoricalSimulator


def main():
    parser = argparse.ArgumentParser(description='Conservative 1-D cut-cell diffusion')
    parser.add_argument('--cells', type=int, default=50)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--dt', type=float, default=0.001, help='maximum requested timestep')
    parser.add_argument('--diffusivity', type=float, default=0.1)
    parser.add_argument('--method', choices=['euler', 'ssprk3'], default='ssprk3')
    parser.add_argument('--lean', action='store_true', help='request concrete Lean budget certificates')
    parser.add_argument('--require-lean', action='store_true', help='fail unless every budget receives a Lean certificate')
    parser.add_argument('--nonnegative', action='store_true', help='reject steps with negative concentrations')
    parser.add_argument('--history-limit', type=int, default=10000)
    parser.add_argument('--output', default='categorical_results.json')
    parser.add_argument('--plot', help='optional plot filename (requires matplotlib)')
    args = parser.parse_args()
    try:
        sim = VerifiedCategoricalSimulator(args.cells, diffusivity=args.diffusivity,
                  dt=args.dt, method=args.method, use_lean=args.lean, require_lean=args.require_lean,
                  history_limit=args.history_limit, require_nonnegative=args.nonnegative)
        sim.run_verified(args.steps)
        sim.save_results(args.output)
        if args.plot:
            sim.visualize(args.plot)
    except (ValueError, ArithmeticError, OSError, RuntimeError) as error:
        parser.exit(1, f'Error: {error}\n')
    print(f'Steps={sim.model.iteration}; elapsed time={sim.model.time:.8g}; '
          f'Lean certificates={sim.lean_certificate_count}; '
          f'results={args.output}')


if __name__ == '__main__':
    main()
