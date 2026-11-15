#!/usr/bin/env bash
# Run the KIF test suite on several Python versions.
#
# Each Python version gets its own venv in .venvs/py<version>, with KIF installed from pyproject.toml
# using the newest dependency versions available for that Python (what a pip user would get).
# uv.lock is not used or changed: its pinned versions don't all have wheels for the newest Pythons.
# For the locked environment, use `uv run pytest` as normal.
#
# Usage:
#   scripts/test_python_versions.sh                     # Python 3.12 to 3.14
#   scripts/test_python_versions.sh --run-slow          # any arguments are passed on to pytest
#   PYTHONS="3.13 3.14" scripts/test_python_versions.sh # choose the Python versions

set -uo pipefail

PYTHONS="${PYTHONS:-3.12 3.13 3.14}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

declare -A RESULTS

for version in $PYTHONS; do
    echo
    echo "=============================== Python $version ==============================="
    venv=".venvs/py$version"

    if ! uv venv --clear --quiet --python "$version" "$venv"; then
        RESULTS[$version]="could not create venv (is Python $version available to uv?)"
        continue
    fi

    if ! uv pip install --quiet --python "$venv" -e ".[tutorials]" pytest; then
        RESULTS[$version]="dependency install failed"
        continue
    fi

    if "$venv/bin/python" -m pytest "$@"; then
        RESULTS[$version]="passed"
    else
        RESULTS[$version]="tests failed"
    fi
done

echo
echo "=============================== Summary ==============================="
exit_code=0
for version in $PYTHONS; do
    echo "Python $version: ${RESULTS[$version]}"
    [[ "${RESULTS[$version]}" == "passed" ]] || exit_code=1
done
exit $exit_code
