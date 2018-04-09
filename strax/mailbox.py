from functools import partial
import heapq
import threading
import logging

from strax.utils import exporter
export, __all__ = exporter()


@export
class MailboxException(Exception):
    pass


@export
class MailboxReadTimeout(MailboxException):
    pass


@export
class MailboxFullTimeout(MailboxException):
    pass


@export
class InvalidMessageNumber(MailboxException):
    pass


@export
class MailBoxAlreadyClosed(MailboxException):
    pass


@export
class OrderedMailbox:
    """A publish-subscribe mailbox, whose subscribers iterate
    over messages set by monotonously incrementing message numbers.
    """

    def __init__(self,
                 name='mailbox',
                 default_send_timeout=20,
                 max_messages=3):
        self.name = name
        self.default_send_timeout = default_send_timeout
        self.max_messages = max_messages

        self.log = logging.getLogger(self.name)
        self.mailbox = []
        self.subscribers_have_read = []
        self.sent_messages = 0
        self.closed = False

        self.lock = threading.RLock()
        self.read_condition = threading.Condition(lock=self.lock)
        self.write_condition = threading.Condition(lock=self.lock)

        self.log.debug("Initialized")

    def send(self, msg, msg_number=None, timeout=None):
        """Send a message.

        If the mailbox is currently full, sleep until there
        is room for your message (or timeout occurs)
        """
        if timeout is None:
            timeout = self.default_send_timeout

        with self.lock:
            if self.closed:
                raise MailBoxAlreadyClosed(f"Can't send to closed {self.name}")

            # We accept int numbers or anything which equals to it's int(...)
            # (like numpy integers)
            if msg_number is None:
                msg_number = self.sent_messages
            try:
                int(msg_number)
                assert msg_number == int(msg_number)
            except (ValueError, AssertionError):
                raise InvalidMessageNumber("Msg numbers must be integers")

            read_until = min(self.subscribers_have_read, default=-1)
            if msg_number <= read_until:
                raise InvalidMessageNumber(
                    f'Attempt to send message {msg_number} while '
                    f'subscribers already read {read_until}.')

            def can_write():
                return len(self.mailbox) < self.max_messages
            if not can_write():
                self.log.debug(f"Mailbox full, wait to send {msg_number}")
            if not self.write_condition.wait_for(can_write, timeout=timeout):
                raise MailboxFullTimeout(f"{self.name} emptied too slow")

            heapq.heappush(self.mailbox, (msg_number, msg))
            self.log.debug(f"Sent {msg_number}")
            self.sent_messages += 1
            self.read_condition.notify_all()

    def close(self, timeout=None):
        with self.lock:
            self.send(StopIteration, timeout=timeout)
            self.closed = True
        self.log.debug(f"Closed to incoming messages")

    def send_from(self, iterable):
        for x in iterable:
            self.send(x)
        self.close()

    def subscribe(self, pass_msg_number=False, timeout=10):
        with self.lock:
            subscriber_i = len(self.subscribers_have_read)
            self.subscribers_have_read.append(-1)
            self.log.debug("Subscribed")
            return self._read(subscriber_i=subscriber_i,
                              pass_msg_number=pass_msg_number,
                              timeout=timeout)

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.name}>"

    def _get_msg(self, number):
        for msg_number, msg in self.mailbox:
            if msg_number == number:
                return msg

    def _has_msg(self, number):
        return any([msg_number == number
                    for msg_number, _ in self.mailbox])

    @property
    def _lowest_msg_number(self):
        return self.mailbox[0][0]

    def _read(self, subscriber_i, pass_msg_number, timeout):
        """Iterate over incoming messages in order.

        Your thread will sleep until the next message is available, or timeout
        expires (in which case MailboxTimeout is raised)
        """
        self.log.debug("Start reading")
        next_number = 0
        last_message = False

        while not last_message:
            with self.lock:
                # Wait until new messages are ready
                next_ready = partial(self._has_msg, next_number)
                if not next_ready():
                    self.log.debug(f"Checking/waiting for {next_number}")
                if not self.read_condition.wait_for(next_ready, timeout):
                    raise MailboxReadTimeout(
                        f"{self.name} did not get {next_number} in time")

                # Grab all messages we can yield
                to_yield = []
                while self._has_msg(next_number):
                    msg = self._get_msg(next_number)
                    if msg is StopIteration:
                        self.log.debug(f"{next_number} is StopIteration")
                        last_message = True
                    to_yield.append((next_number, msg))
                    next_number += 1

                if len(to_yield) > 1:
                    self.log.debug(f"Read {to_yield[0][0]}-{to_yield[-1][0]}")
                else:
                    self.log.debug(f"Read {to_yield[0][0]}")

                self.subscribers_have_read[subscriber_i] = next_number - 1

                # Clean up the mailbox
                while (len(self.mailbox)
                       and (min(self.subscribers_have_read)
                            >= self._lowest_msg_number)):
                    heapq.heappop(self.mailbox)
                self.write_condition.notify_all()

            for msg_number, msg in to_yield:
                if msg is StopIteration:
                    return
                if pass_msg_number:
                    yield msg_number, msg
                else:
                    yield msg

        self.log.debug("Done reading")
