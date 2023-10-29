from .base import *
from .threaded_mailbox import *

PROCESSORS = {
    "default": ThreadedMailboxProcessor,
    "threaded_mailbox": ThreadedMailboxProcessor,
}
