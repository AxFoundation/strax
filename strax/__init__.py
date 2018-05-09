# flake8: noqa
__version__ = '0.1.2'

# Glue the package together
# See https://www.youtube.com/watch?v=0oTh1CXRaQ0 if this confuses you
# The order of subpackes is not invariant, since we use strax.xxx inside strax
from .utils import *
from .dtypes import *
from strax.processing.general import *
from .chunk_arrays import *

from .storage import *
from .plugin import *
from .mailbox import *
from .processor import *
from .core import *

from .io import *

from strax.processing.data_reduction import *
from strax.processing.pulse_processing import *
from strax.processing.peak_building import *
from strax.processing.peak_splitting import *
from strax.processing.peak_properties import *

from . import xenon
