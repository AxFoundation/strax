# flake8: noqa
__version__ = '0.7.5'

# Glue the package together
# See https://www.youtube.com/watch?v=0oTh1CXRaQ0 if this confuses you
# The order of subpackes is not invariant, since we use strax.xxx inside strax
from .utils import *
from .dtypes import *
from strax.processing.general import *
from .chunk_arrays import *

from .storage.common import *
from .storage.files import *
from .storage.mongo import *
from .storage.s3 import *
from .storage.zipfiles import *

from .config import *
from .plugin import *
from .mailbox import *
from .processor import *
from .context import *

# Just run this file, it will add new methods to Context
from . import run_selection
del run_selection

from .io import *

from strax.processing.data_reduction import *
from strax.processing.pulse_processing import *
from strax.processing.peak_building import *
from strax.processing.peak_splitting import *
from strax.processing.peak_properties import *
