import heapq
import threading

__all__ = ('MailboxReadTimeout', 'MailboxFullTimeout',
           'InvalidMessageNumber', 'OrderedMailbox')


class MailboxReadTimeout(Exception):
    pass


class MailboxFullTimeout(Exception):
    pass


class InvalidMessageNumber(Exception):
    pass


class OrderedMailbox:
    """A publish-subscribe mailbox, whose subscribers iterate
    over messages set by monotonously incrementing message numbers.
    """

    def __init__(self, max_messages=float('inf')):
        self.mailbox = []
        self.subscribers_have_read = []
        self.sent_messages = 0
        self.max_messages = max_messages
        self.read_condition = threading.Condition()
        self.write_condition = threading.Condition()

    def send(self, msg, number=None, timeout=None):
        """Send a message.

        If the mailbox is currently full, sleep until there
        is room for your message (or timeout occurs)
        """
        # Determine / validate the message number
        if number is None:
            number = self.sent_messages
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
            self.sent_messages += 1

        with self.read_condition:
            self.read_condition.notify_all()

    def subscribe(self, pass_msg_number=False, timeout=None):
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
                    print(f"Subscriber {subscriber_i} grabbed {msg_number}")
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

            for x in to_yield:
                if x[1] is StopIteration:
                    return
                if pass_msg_number:
                    yield x
                else:
                    yield x[1]
