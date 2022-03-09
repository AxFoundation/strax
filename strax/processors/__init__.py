from .base import ProcessorComponents, BaseProcessor
from .threaded_mailbox_processor import ThreadedMailboxProcessor

PROCESSORS = {
    "default": ThreadedMailboxProcessor,
    "threaded_mailbox_processor": ThreadedMailboxProcessor
}

# FIXME: add entrypoint logic for processor plugins.