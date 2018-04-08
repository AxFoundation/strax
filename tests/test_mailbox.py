import concurrent.futures
import threading
import time

import numpy as np
import pytest

from strax import OrderedMailbox, MailboxReadTimeout, MailboxFullTimeout

SHORT_TIMEOUT = 0.1
LONG_TIMEOUT = 5 * SHORT_TIMEOUT


def reader(source, reader_sleeps=0, name=''):
    result = []
    for x in source:
        print(f"Reader {name} got {x}, sleeping for {reader_sleeps}")
        time.sleep(reader_sleeps)
        print(f"Reader {name} awoke")
        result.append(x)
    return result

def mailbox_tester(messages,
                   numbers=None,
                   reader_sleeps=0.,
                   max_messages=100,
                   timeout=SHORT_TIMEOUT,
                   result_timeout=LONG_TIMEOUT):
    if numbers is None:
        numbers = np.arange(len(messages))

    mb = OrderedMailbox(max_messages=max_messages)

    n_readers = 2
    with concurrent.futures.ThreadPoolExecutor() as tp:
        futures = [tp.submit(reader,
                             source=mb.subscribe(timeout=timeout),
                             reader_sleeps=reader_sleeps)
                   for i in range(n_readers)]

        for i in range(len(messages)):
            mb.send(messages[i], msg_number=numbers[i], timeout=timeout)
            print(f"Sent message {i}. Now {len(mb.mailbox)} ms in mailbox.")

        mb.close()

        # Results must be equal
        for f in futures:
            np.testing.assert_equal(f.result(timeout=result_timeout),
                                    messages[np.argsort(numbers)])


def test_result_timeout():
    """Test that our mailbox tester actually times out.
    (if not, the other tests might hang indefinitely if something is broken)
    """
    with pytest.raises(concurrent.futures.TimeoutError):
        mailbox_tester([0, 1], numbers=[1, 2], timeout=2 * LONG_TIMEOUT)


def test_read_timeout():
    """Subscribers time out if we cannot read for too long"""
    with pytest.raises(MailboxReadTimeout):
        mailbox_tester([0, 1], numbers=[1, 2])


def test_write_timeout():
    """Writers time out if we cannot write for too long"""
    with pytest.raises(MailboxFullTimeout):
        mailbox_tester([0, 1, 2, 3, 4],
                       max_messages=1,
                       reader_sleeps=LONG_TIMEOUT)


def test_reversed():
    """Mailbox sorts messages properly"""
    mailbox_tester(np.arange(10), numbers=np.arange(10)[::-1])


def test_deadlock_regression():
    """A reader thread may start after the first message is processed"""
    mb = OrderedMailbox()
    mb.send(0)

    readers = [
        threading.Thread(target=reader,
                         kwargs=dict(
                             source=mb.subscribe(timeout=SHORT_TIMEOUT),
                             name=str(i)))
        for i in range(2)
    ]
    readers[0].start()
    time.sleep(SHORT_TIMEOUT)

    readers[1].start()
    mb.send(1)
    mb.close()

    for t in readers:
        t.join(SHORT_TIMEOUT)
        assert not t.is_alive()
