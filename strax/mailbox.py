from concurrent.futures import Future, TimeoutError
import heapq
import sys
import threading
import typing
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
class MailboxKilled(MailboxException):
    pass


@export
class Mailbox:
    """Publish/subscribe mailbox for builing complex pipelines
    out of simple iterators, using multithreading.

    A sender can be any iterable. To read from the mailbox, either:
     1. Use .subscribe() to get an iterator.
        You can only use one of these per thread.
     2. Use .add_subscriber(f) to subscribe the function f.
        f should take an iterator as its first argument
        (and actually iterate over it, of course).

    Each sender and receiver is wrapped in a thread, so they can be paused:
     - senders, if the mailbox is full;
     - readers, if they call next() but the next message is not yet available.

    Any futures sent in are awaited before they are passed to receivers.

    Exceptions in a sender cause MailboxKilled to be raised in each reader.
    If the reader doesn't catch this, and it writes to another mailbox,
    this therefore kills that mailbox (raises MailboxKilled for each reader)
    as well. Thus MailboxKilled exceptions travel downstream in pipelines.

    Sender threads are not killed by exceptions raise in readers.
    To kill sender threads too, use .kill(upstream=True). Even this does not
    propagate further upstream than the immediate sender threads.
    """

    DEFAULT_TIMEOUT = 300
    DEFAULT_MAX_MESSAGES = 2

    def __init__(self,
                 name='mailbox',
                 timeout=None,
                 max_messages=None):
        self.name = name
        if timeout is None:
            timeout = self.DEFAULT_TIMEOUT
        self.timeout = timeout
        if max_messages is None:
            max_messages = self.DEFAULT_MAX_MESSAGES
        self.max_messages = max_messages

        self.closed = False
        self.force_killed = False
        self.killed = False
        self.killed_because = None

        self._mailbox = []
        self._subscribers_have_read = []
        self._n_sent = 0
        self._threads = []
        self._lock = threading.RLock()
        self._read_condition = threading.Condition(lock=self._lock)
        self._write_condition = threading.Condition(lock=self._lock)

        self.log = logging.getLogger(self.name)
        self.log.debug("Initialized")

    def add_sender(self, source, name=None):
        """Configure mailbox to read from an iterable source

        :param source: Iterable to read from
        :param name: Name of the thread in which the function will run.
            Defaults to source:<mailbox_name>
        """
        if name is None:
            name = f'source:{self.name}'
        t = threading.Thread(target=self._send_from,
                             name=name,
                             args=(source,))
        self._threads.append(t)

    def add_reader(self, subscriber, name=None, **kwargs):
        """Subscribe a function to the mailbox.

        Function should accept the generator over messages as first argument.
        kwargs will be passed to function.

        :param name: Name of the thread in which the function will run.
            Defaults to read_<number>:<mailbox_name>
        """
        if name is None:
            name = f'read_{self._n_subscribers}:{self.name}'
        t = threading.Thread(target=subscriber,
                             name=name,
                             args=(self.subscribe(),),
                             kwargs=kwargs)
        self._threads.append(t)

    def subscribe(self):
        """Return generator over messages in the mailbox
        """
        with self._lock:
            subscriber_i = self._n_subscribers
            self._subscribers_have_read.append(-1)
            self.log.debug(f"Added subscriber {subscriber_i}")
            return self._read(subscriber_i=subscriber_i)

    def start(self):
        if not self._n_subscribers:
            raise ValueError(f"Attempt to start mailbox {self.name} "
                             f"without subscribers")
        for t in self._threads:
            t.start()

    def kill_from_exception(self, e, reraise=True):
        """Kill the mailbox following a caught exception e"""
        if isinstance(e, MailboxKilled):
            # Kill this mailbox too.
            self.log.debug("Propagating MailboxKilled exception")
            self.kill(reason=e.args[0])
            # Do NOT raise! One traceback on the screen is enough.
        else:
            self.log.debug(f"Killing mailbox due to exception {e}!")
            self.kill(reason=(e.__class__, e, sys.exc_info()[2]))
            if reraise:
                raise e

    def kill(self, upstream=True, reason=None):
        with self._lock:
            self.log.debug(f"Kill received by {self.name}")
            if upstream:
                self.force_killed = True
            if self.killed:
                self.log.debug(f"Double kill on {self.name} = NOP")
                return
            self.killed = True
            self.killed_because = reason
            self._read_condition.notify_all()
            self._write_condition.notify_all()

    def cleanup(self):
        for t in self._threads:
            t.join(timeout=self.timeout)
            if t.is_alive():
                raise RuntimeError("Thread %s did not terminate!" % t.name)

    def _send_from(self, iterable):
        """Send to mailbox from iterable, exiting appropriately if an
        exception is thrown
        """
        try:
            for x in iterable:
                self.send(x)
        except Exception as e:
            self.kill_from_exception(e)
        else:
            self.log.debug("Producing iterable exhausted, regular stop")
            self.close()

    def send(self, msg, msg_number: typing.Union[int, None] = None):
        """Send a message.

        If the message is a future, receivers will be passed its result.
        (possibly waiting for completion if needed)

        If the mailbox is currently full, sleep until there
        is room for your message (or timeout occurs)
        """
        with self._lock:
            if self.closed:
                raise MailBoxAlreadyClosed(f"Can't send to closed {self.name}")
            if self.force_killed:
                self.log.debug(f"Sender found {self.name} force-killed")
                raise MailboxKilled(self.killed_because)
            if self.killed:
                self.log.debug("Send to killed mailbox: message lost")
                return

            # We accept int numbers or anything which equals to it's int(...)
            # (like numpy integers)
            if msg_number is None:
                msg_number = self._n_sent
            try:
                int(msg_number)
                assert msg_number == int(msg_number)
            except (ValueError, AssertionError):
                raise InvalidMessageNumber("Msg numbers must be integers")

            read_until = min(self._subscribers_have_read, default=-1)
            if msg_number <= read_until:
                raise InvalidMessageNumber(
                    f'Attempt to send message {msg_number} while '
                    f'subscribers already read {read_until}.')

            def can_write():
                return (len(self._mailbox) < self.max_messages or self.killed)
            if not can_write():
                self.log.debug("Subscribers have read: "
                               + str(self._subscribers_have_read))
                self.log.debug(f"Mailbox full, wait to send {msg_number}")
            if not self._write_condition.wait_for(can_write,
                                                  timeout=self.timeout):
                raise MailboxFullTimeout(f"{self.name} emptied too slow")

            if self.killed:
                self.log.debug(f"Sender found {self.name} killed while waiting"
                               " for room for new messages.")
                # TODO: this is duplicated from above...
                if self.force_killed:
                    raise MailboxKilled(self.killed_because)
                return

            heapq.heappush(self._mailbox, (msg_number, msg))
            self.log.debug(f"Sent {msg_number}")
            self._n_sent += 1
            self._read_condition.notify_all()

    def close(self):
        self.log.debug(f"Closing; sending StopIteration")
        with self._lock:
            self.send(StopIteration)
            self.closed = True
        self.log.debug(f"Closed to incoming messages")

    def _read(self, subscriber_i):
        """Iterate over incoming messages in order.

        Your thread will sleep until the next message is available, or timeout
        expires (in which case MailboxReadTimeout is raised)
        """
        self.log.debug("Start reading")
        next_number = 0
        last_message = False

        while not last_message:
            with self._lock:

                # Wait for new messages
                def next_ready():
                    return self._has_msg(next_number) or self.killed
                if not next_ready():
                    self.log.debug(f"Checking/waiting for {next_number}")
                if not self._read_condition.wait_for(next_ready,
                                                     self.timeout):
                    raise MailboxReadTimeout(
                        f"{self.name} did not get {next_number} in time")

                if self.killed:
                    self.log.debug(f"Reader finds {self.name} killed")
                    raise MailboxKilled(self.killed_because)

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
                    self.log.debug(f"Read {to_yield[0][0]}-{to_yield[-1][0]}  in subscriber {subscriber_i}")
                else:
                    self.log.debug(f"Read {to_yield[0][0]} in subscriber {subscriber_i}")

                self._subscribers_have_read[subscriber_i] = next_number - 1

                # Clean up the mailbox
                while (len(self._mailbox)
                       and (min(self._subscribers_have_read)
                            >= self._lowest_msg_number)):
                    heapq.heappop(self._mailbox)
                self._write_condition.notify_all()

            for msg_number, msg in to_yield:
                if msg is StopIteration:
                    break
                elif isinstance(msg, Future):
                    if not msg.done():
                        self.log.debug(f"Waiting for future {msg_number}")
                        try:
                            res = msg.result(timeout=self.timeout)
                        except TimeoutError:
                            raise TimeoutError(
                                f"Future {msg_number} timed out!")
                        self.log.debug(f"Future {msg_number} completed")
                    else:
                        res = msg.result()
                        self.log.debug(f"Future {msg_number} was already done")
                else:
                    res = msg

                try:
                    yield res
                except Exception as e:
                    # TODO: Should I also handle timeout errors like this?
                    self.kill_from_exception(e)

        self.log.debug("Done reading")

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.name}>"

    def _get_msg(self, number):
        for msg_number, msg in self._mailbox:
            if msg_number == number:
                return msg
        else:
            raise RuntimeError(f"Could not find message {number}")

    def _has_msg(self, number):
        """Return if mailbox has message number.

        Also returns True if mailbox is killed, so be sure to check
        self.killed after this!
        """
        if self.killed:
            return True
        return any([msg_number == number
                    for msg_number, _ in self._mailbox])

    @property
    def _n_subscribers(self):
        return len(self._subscribers_have_read)

    @property
    def _lowest_msg_number(self):
        return self._mailbox[0][0]


@export
def divide_outputs(source, mailboxes, outputs=None):
    """This code is a 'mail sorter' which gets dicts of arrays from source
    and sends the right array to the right mailbox.
    """
    # raise ZeroDivisionError   # TODO: check this is handled properly
    if outputs is None:
        outputs = mailboxes.keys()
    mbs_to_kill = [mailboxes[d] for d in outputs]
    # TODO: this code duplicates exception handling and cleanup
    # from Mailbox.send_from! Can we avoid that somehow?
    try:
        for result in source:
            for d, x in result.items():
                mailboxes[d].send(x)
    except Exception as e:
        for m in mbs_to_kill:
            m.kill_from_exception(e, reraise=False)
        if not isinstance(e, MailboxKilled):
            raise
    else:
        for m in mbs_to_kill:
            m.close()
