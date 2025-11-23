#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def run_main(verbose=False):
    repo_root = Path(__file__).resolve().parent
    main_py = repo_root / 'main.py'
    if not main_py.exists():
        print("Error: main.py not found in project root.", file=sys.stderr)
        return 2
    cmd = [sys.executable, str(main_py)]
    if verbose:
        print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd)
    return proc.returncode


def run_dashboard(verbose=False):
    repo_root = Path(__file__).resolve().parent
    ui_py = repo_root / 'ui.py'
    if not ui_py.exists():
        print("Error: ui.py not found in project root.", file=sys.stderr)
        return 2
    cmd = [sys.executable, str(ui_py)]
    if verbose:
        print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd)
    return proc.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(prog='astra', description='Astra CLI - project helper')
    parser.add_argument('--sim', action='store_true', help='Run simulation (execute main.py)')
    parser.add_argument('--dashboard', action='store_true', help='Run dashboard UI (execute ui.py)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args(argv)

    if args.sim:
        rc = run_main(verbose=args.verbose)
        sys.exit(rc)
    if args.dashboard:
        rc = run_dashboard(verbose=args.verbose)
        sys.exit(rc)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
