"""Unit tests for the contact_identification helper fns"""

import numpy as np
import pytest
from MDAnalysis import Universe
from MDAnalysis.tests.datafiles import DCD, PSF

from key_interactions_finder import contact_identification as ci


@pytest.fixture(scope="module")
def psf_universe():
    return Universe(PSF, DCD)


def test_score_residue_contact_known_values():
    # PyContact scoring function: 1 / (1 + exp(5 * (d - 4))) for each atom pair with d <= cut-off.
    # d = 4 -> 0.5 exactly.
    assert ci._score_residue_contact(np.array([[4.0]])) == 0.5
    # Two atom pairs at 4 A -> 1.0.
    assert ci._score_residue_contact(np.array([[4.0, 4.0]])) == 1.0
    # Very close contact -> ~1 per pair.
    assert ci._score_residue_contact(np.array([[2.0]])) == pytest.approx(1.0, abs=1e-4)


def test_score_residue_contact_ignores_pairs_beyond_cutoff():
    assert ci._score_residue_contact(np.array([[6.01, 10.0]])) == 0.0
    assert ci._score_residue_contact(np.array([[5.0]]), dist_cut=5.0) > 0.0


def test_score_residue_contact_matches_vectorised_formula():
    rng = np.random.default_rng(0)
    dists = rng.uniform(2.0, 8.0, size=(7, 9))
    expected = np.where(dists <= 6.0, 1 / (1 + np.exp(5 * (dists - 4))), 0).sum()
    assert ci._score_residue_contact(dists) == pytest.approx(round(expected, 4), abs=1e-4)


def test_determine_interaction_type_salt_bridge(psf_universe):
    # Use the first LYS/ARG and first GLU/ASP in the topology.
    resnames = psf_universe.residues.resnames
    pos = int(np.where(np.isin(resnames, ["LYS", "ARG"]))[0][0]) + 1
    neg = int(np.where(np.isin(resnames, ["GLU", "ASP"]))[0][0]) + 1
    assert ci._determine_interaction_type(pos, neg, [], psf_universe) == "Saltbr"
    assert ci._determine_interaction_type(neg, pos, [], psf_universe) == "Saltbr"


def test_determine_interaction_type_hbond_either_order(psf_universe):
    # Pick two non-charged, non-hydrophobic residues so only the H-bond rule can apply.
    resnames = list(psf_universe.residues.resnames)
    polar = [i + 1 for i, name in enumerate(resnames) if name in ("SER", "THR", "ASN", "GLN")]
    res1, res2 = polar[0], polar[1]
    assert ci._determine_interaction_type(res1, res2, [(res1, res2)], psf_universe) == "Hbond"
    assert ci._determine_interaction_type(res1, res2, [(res2, res1)], psf_universe) == "Hbond"
    assert ci._determine_interaction_type(res1, res2, [], psf_universe) == "Other"


def test_atom_num_to_res_info(psf_universe):
    atom = psf_universe.residues[4].atoms[0]
    assert ci._atom_num_to_res_info(atom.index, psf_universe) == (atom.resname, atom.resid)
