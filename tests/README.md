# Running the Tests

KIF has a [pytest](https://docs.pytest.org/) test suite in the `tests/` folder. Run it from the repository root with [uv](https://docs.astral.sh/uv/).

## Setup

```bash
git clone https://github.com/kamerlinlab/KIF.git
cd KIF
uv sync --dev
```

## Running the tests

```bash
uv run pytest

# Only run the unit tests (a few seconds), or only the end-to-end tests (minutes).
uv run pytest tests/unit
uv run pytest tests/e2e

# Also run the slow tests in tests/slow/ (about 2 more minutes, plus a 90 MB download the first time).
uv run pytest --run-slow
```

Without `--run-slow`, the slow tests are reported as skipped.

## Testing several Python versions

`scripts/test_python_versions.sh` runs the test suite on Python 3.12 to 3.14 (the supported versions), each in its own venv in `.venvs/`. KIF is installed with the newest dependency versions available for each Python (as a `pip install KIF` user would get), not the versions pinned in `uv.lock`. 

```bash
scripts/test_python_versions.sh
scripts/test_python_versions.sh --run-slow           # run the slow tutorial tests. 
PYTHONS="3.13 3.14" scripts/test_python_versions.sh  # test a subset of versions.
```

## Test data

`tests/data/` holds the committed test data:

- `ptp1b_contacts_subset.csv`, `ptp1b_classes_subset.txt`, `ptp1b_closed.pdb`: every 100th frame of the PTP1B tutorial dataset, with the matching class labels and the closed-state structure.
- `contacts_psf_dcd_10_frames_expected.csv`: the expected `calculate_contacts` output for the first 10 frames of the MDAnalysis PSF/DCD test system.

These are made by `tests/make_test_data.py`. Only regenerate them when you intend to change the output, and check the git diff afterwards:

```bash
uv run python -m tests.make_test_data
```
The tests and linting also runs on GitHub Actions (`.github/workflows/tests.yml`).
