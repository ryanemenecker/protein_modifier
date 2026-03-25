"""A python package for manipulating protein structures"""

# Add imports here
from .modify import *

from ._version import __version__

# Configure package-level logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
