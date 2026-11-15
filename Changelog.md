## Change Log

0.5.0 (15/11/2026)

---

#### Changed:

- Added a test suite and a GitHub Actions workflow that runs it on PR to main and push to main.
- Minor bug fixes and code updates to work with newer versions of dependencies.
- Used hatchling to build the package. 
- Supported Python versions are now 3.12 to 3.14.
- The packages only needed for the tutorials are now an optional extra: `pip install "KIF[tutorials]"`.

0.4.2 (31/05/2025)

---

#### Added:

- Swapped from outdated setup.py to pyproject.toml
- Went over codebase with ruff.
- Using UV instead of pip/venv.

  0.4.1 (30/05/2024)

---

#### Fixed:

- Added new tutorials files to allowed google drive links

  0.4.0 (30/05/2024)

---

#### Added:

Now KIF results can be projected onto a given 3D structure using [ChimeraX](https://www.cgl.ucsf.edu/chimerax/). Previously only [PyMOL](https://pymol.org/) was possible. Thanks to @darianyang for adding this functionality.

0.3.4 (13/02/2024)

---

#### Fixed:

- Remove bug from "estimate_feature_directions" function.

Between 0.3.1 to 0.3.3 (04/10/2023)

---

#### Updated:

- Changed pymol representation method to no longer depend on external dependancy draw_links.py.
- Able to recognise/handle (non-standard) residues that have one or more numerical character in their name.

#### Fixed:

- Block wise analysis correctly treated in contact_identification.py

  0.3.0 (14/07/2023)

---

#### Updated:

- Large scale speed up of contact analysis module.

  0.2.0 (05/04/2023)

---

#### Added:

- New method to calculate all contacts in a simulation provided within KIF.
  Avoids the need to install/use PyContact which is no longer actively maintained.

#### Updated:

- Datasets can now optionally contain information about the side chain or main chain.
  Before, an error could have occured if datasets did not have this information.

  0.1.1 (30/03/2023)

---

#### Updated:

- Some minor linting/reformatting of code.

#### Fixed:

- Handle evaluation of negative target variable values for ML regression.

  0.1.0 (15/12/2022)

---

Initial release
