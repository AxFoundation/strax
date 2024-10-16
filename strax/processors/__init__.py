from .base import *
from .threaded_mailbox import *
from .single_thread import *

# This is redundant with the star-imports above, but some flake8
# versions require this
from .threaded_mailbox import ThreadedMailboxProcessor
from .single_thread import SingleThreadProcessor

PROCESSORS = {
    "default": SingleThreadProcessor,
    "threaded_mailbox": ThreadedMailboxProcessor,
    "single_thread": SingleThreadProcessor,
}
