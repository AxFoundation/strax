from .base import *
from .threaded_mailbox import *
from .single_thread import *

PROCESSORS = {
    "default": ThreadedMailboxProcessor,
    "threaded_mailbox": ThreadedMailboxProcessor,
    "single_thread": SingleThreadProcessor,
}
