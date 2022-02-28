"""
Test the pycontact file processing module.
"""
import pandas as pd
import numpy as np
import pytest
from key_interactions_finder import pycontact_processing


def test_add_can_add_numbers():
    # given
    num = 3
    num2 = 45

    # when
    result = num + num2

    # then
    assert result == 48


def test_template():
    """

    """
    # given
    #num = 3
    #num2 = 45

    # when
    #result = num + num2

    # then
    # assert
