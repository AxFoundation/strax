from .base import *
from .threaded_mailbox_processor import *

PROCESSORS = {
    "default": ThreadedMailboxProcessor,
    "threaded_mailbox_processor": ThreadedMailboxProcessor
}

# FIXME: add entrypoint logic for processor plugins.