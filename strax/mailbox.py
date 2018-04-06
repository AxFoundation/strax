import heapq
import threading
import logging

from strax.utils import exporter, setup_logger
export, __all__ = exporter()


@export
class MailboxReadTimeout(Exception):
    pass


@export
class MailboxFullTimeout(Exception):
    pass


@export
class InvalidMessageNumber(Exception):
    pass


@export
class OrderedMailbox:
    """A publish-subscribe mailbox, whose subscribers iterate
    over messages set by monotonously incrementing message numbers.
    """

    def __init__(self, name='mailbox', max_messages=float('inf')):
        self.name = name
        self.log = setup_logger(self.name)

        self.mailbox = []
        self.subscribers_have_read = []
        self.sent_messages = 0
        self.max_messages = max_messages
        self.read_condition = threading.Condition()
        self.write_condition = threading.Condition()

        self.log.debug("Initialized")

    def send(self, msg, number=None, timeout=30):
        """Send a message.

        If the mailbox is currently full, sleep until there
        is room for your message (or timeout occurs)
        """
        # Determine / validate the message number
        if number is None:
            number = self.sent_messages
        if not len(self.subscribers_have_read):
            read_until = -1
        else:
            read_until = min(self.subscribers_have_read)
        if number <= read_until:
            raise InvalidMessageNumber(
                f'Attempt to send message {number} while '
                f'subscribers already read {read_until}.')

        def can_write():
            return len(self.mailbox) < self.max_messages

        with self.write_condition:
            if not can_write():
                if not self.write_condition.wait_for(can_write,
                                                     timeout=timeout):
                    raise MailboxFullTimeout

            heapq.heappush(self.mailbox, (number, msg))
            self.log.debug(f"Wrote {number}")
            self.sent_messages += 1

        with self.read_condition:
            self.read_condition.notify_all()

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.name}>"

    def close(self):
        self.log.debug(f"Closing")
        self.send(StopIteration)

    def subscribe(self, pass_msg_number=False, timeout=30):
        """Iterate over incoming messages in order.

        Your thread will sleep until the next message is available, or timeout
        expires (in which case MailboxTimeout is raised)
        """
        subscriber_i = len(self.subscribers_have_read)
        self.subscribers_have_read.append(-1)

        next_number = 0
        last_message = False

        while not last_message:
            with self.read_condition:
                # Wait until new messages are ready
                def can_read():
                    return (len(self.mailbox)
                            and self.mailbox[0][0] == next_number)
                if not self.read_condition.wait_for(can_read, timeout):
                    raise MailboxReadTimeout

                # Grab all messages we can yield
                to_yield = []
                for msg_number, msg in self.mailbox:

                    if msg_number > next_number:
                        break
                    self.log.debug(f"Read {msg_number}")
                    if msg is StopIteration:
                        last_message = True
                    to_yield.append((msg_number, msg))
                    next_number += 1

                self.subscribers_have_read[subscriber_i] = next_number - 1

                # Clean up the mailbox
                with self.write_condition:
                    while (len(self.mailbox)
                           and min(self.subscribers_have_read)
                           >= self.mailbox[0][0]):
                        heapq.heappop(self.mailbox)
                    self.write_condition.notify_all()

            for msg_number, msg in to_yield:
                if msg is StopIteration:
                    return
                if pass_msg_number:
                    yield msg_number, msg
                else:
                    yield msg
