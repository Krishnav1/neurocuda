#!/usr/bin/env python3
"""
Publish NeuroCUDA to PyPI (The Python Package Index).

Usage:
    python scripts/publish.py --test          # Publish to TestPyPI (safe, for verification)
    python scripts/publish.py                 # Publish to PyPI (live — pip install neurocuda)

Prerequisites:
    1. Create an account: https://pypi.org/account/register/
    2. Create API token:  https://pypi.org/manage/account/token/
    3. Set token:         export TWINE_USERNAME=__token__
                          export TWINE_PASSWORD=pypi-xxxxxxxx
    Or use keyring:       pip install keyring
                          twine register --username __token__ --password pypi-xxxx

This script:
    1. Cleans previous builds
    2. Builds the wheel (and source distribution)
    3. Checks the package with twine
    4. Uploads to PyPI or TestPyPI
"""

import sys, os, subprocess, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


def run(cmd, desc):
    print(f"\n  [{desc}]")
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:500]}")
        sys.exit(1)
    # Print last few lines of output
    lines = result.stdout.strip().split('\n')
    for line in lines[-5:]:
        print(f"    {line}")
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description="Publish NeuroCUDA to PyPI")
    parser.add_argument("--test", action="store_true",
                       help="Publish to TestPyPI instead of real PyPI")
    parser.add_argument("--skip-check", action="store_true",
                       help="Skip twine check (faster iteration)")
    args = parser.parse_args()

    target = "TestPyPI" if args.test else "PyPI"
    print("=" * 60)
    print(f"  Publishing NeuroCUDA to {target}")
    print("=" * 60)

    # 1. Clean
    run("rm -rf dist build *.egg-info neurocuda.egg-info", "Cleaning old builds")

    # 2. Build
    run("python -m build", "Building wheel + sdist")

    # 3. Show what was built
    run("ls -lh dist/", "Built artifacts")

    # 4. Check
    if not args.skip_check:
        run("twine check dist/*", "Twine check")

    # 5. Upload
    if args.test:
        upload_cmd = "twine upload --repository testpypi dist/*"
    else:
        upload_cmd = "twine upload dist/*"

    print(f"\n  {'='*60}")
    print(f"  Ready to upload to {target}")
    print(f"  {'='*60}")
    print(f"\n  Command: {upload_cmd}")
    print(f"\n  If this is your first time:")
    print(f"    1. pip install twine")
    print(f"    2. Set TWINE_USERNAME=__token__")
    print(f"    3. Set TWINE_PASSWORD=pypi-xxxxxxxx")
    print(f"\n  Or create ~/.pypirc:")
    print(f"    [pypi]")
    print(f"    username = __token__")
    print(f"    password = pypi-xxxxxxxx")

    # Ask confirmation for real PyPI
    if not args.test:
        confirm = input(f"\n  Upload to LIVE PyPI? (type 'neurocuda' to confirm): ")
        if confirm != "neurocuda":
            print("  Aborted.")
            return 0

    run(upload_cmd, f"Uploading to {target}")

    print(f"\n  ✅ Published to {target}!")
    if args.test:
        print(f"  Install: pip install --index-url https://test.pypi.org/simple/ neurocuda")
    else:
        print(f"  Install: pip install neurocuda")
    print()


if __name__ == "__main__":
    main()
