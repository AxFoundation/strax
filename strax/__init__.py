# flake8: noqa
__version__ = '0.0.1'

# Glue the package together
# See https://www.youtube.com/watch?v=0oTh1CXRaQ0 if this confuses you
# The order of subpackes is not invariant, since we use strax.*** inside strax
from . import io_chunked
from . import chunk_arrays

from .utils import *
from .cache import *
from .core import *

from .dtypes import *
from .mailbox import *
from .io import *
from .plugin import *

from strax.processing.data_reduction import *
from strax.processing.pulse_processing import *
from strax.processing.peak_building import *
from strax.processing.peak_splitting import *
from strax.processing.peak_properties import *

from strax.external import pax_interface, daq_interface
