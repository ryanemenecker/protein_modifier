"""
Unit and regression test for the protein_modifier package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import protein_modifier


def test_protein_modifier_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "protein_modifier" in sys.modules
