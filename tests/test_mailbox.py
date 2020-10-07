import concurrent.futures
import threading
import time

import numpy as np
import pytest

import strax

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
                   lazy=False,
                   reader_sleeps=0.,
                   max_messages=100,
                   expected_result=None,
                   timeout=SHORT_TIMEOUT,
                   result_timeout=LONG_TIMEOUT):
    if numbers is None:
        numbers = np.arange(len(messages))
    if expected_result is None:
        messages = np.asarray(messages)
        expected_result = messages[np.argsort(numbers)]

    mb = strax.Mailbox(max_messages=max_messages,
                       timeout=timeout,
                       lazy=lazy)

    n_readers = 2

    with concurrent.futures.ThreadPoolExecutor() as tp:
        futures = [tp.submit(reader,
                             source=mb.subscribe(),
                             reader_sleeps=reader_sleeps)
                   for _ in range(n_readers)]

        for i in range(len(messages)):
            mb.send(messages[i], msg_number=numbers[i])
            print(f"Sent message {i}. Now {len(mb._mailbox)} ms in mailbox.")

        mb.close()

        # Results must be equal
        for f in futures:
            np.testing.assert_equal(f.result(timeout=result_timeout),
                                    expected_result)


def test_highlevel():
    """Test highlevel mailbox API"""
    for lazy in [False, True]:
        n_threads_start = len(threading.enumerate())
        print(f"Lazy mode: {lazy}")

        mb = strax.Mailbox(lazy=lazy)
        mb.add_sender(iter(list(range(10))))

        def test_reader(source):
            test_reader.got = r = []
            for s in source:
                r.append(s)

        mb.add_reader(test_reader)
        mb.start()
        time.sleep(SHORT_TIMEOUT)
        assert hasattr(test_reader, 'got')
        assert test_reader.got == list(range(10))
        mb.cleanup()
        threads = [f'{t.name} is dead: {True^t.is_alive()}'
                   for t in threading.enumerate()]
        assert len(threads) == n_threads_start, (
            f"Not all threads died. \n Threads running are:{threads}")


def test_result_timeout():
    """Test that our mailbox tester actually times out.
    (if not, the other tests might hang indefinitely if something is broken)
    """
    with pytest.raises(concurrent.futures.TimeoutError):
        mailbox_tester([0, 1],
                       numbers=[1, 2],
                       timeout=2 * LONG_TIMEOUT)


def test_read_timeout():
    """Subscribers time out if we cannot read for too long"""
    with pytest.raises(strax.MailboxReadTimeout):
        mailbox_tester([0, 1], numbers=[1, 2])


def test_write_timeout():
    """Writers time out if we cannot write for too long"""
    with pytest.raises(strax.MailboxFullTimeout):
        mailbox_tester([0, 1, 2, 3, 4],
                       max_messages=1,
                       reader_sleeps=LONG_TIMEOUT)


def test_reversed():
    """Mailbox sorts messages properly"""
    mailbox_tester(np.arange(10),
                   numbers=np.arange(10)[::-1])


def test_deadlock_regression():
    """A reader thread may start after the first message is processed"""
    # Test cannot run in lazy mode, cannot send without active subscriber
    mb = strax.Mailbox(timeout=SHORT_TIMEOUT)
    mb.send(0)

    readers = [
        threading.Thread(target=reader,
                         kwargs=dict(
                             source=mb.subscribe(),
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


def test_close_protection():
    """Cannot send messages to a closed mailbox"""
    mb = strax.Mailbox()
    mb.close()
    with pytest.raises(strax.MailBoxAlreadyClosed):
        mb.send(0)


def test_valid_msg_number():
    """Message numbers are non-negative integers"""
    mb = strax.Mailbox()
    with pytest.raises(strax.InvalidMessageNumber):
        mb.send(0, msg_number=-1)
    with pytest.raises(strax.InvalidMessageNumber):
        mb.send(0, msg_number='???')


# Task for in the next test, must be global since we're using ProcessPool
# (which must pickle)
def _task(i):
    time.sleep(SHORT_TIMEOUT)
    return i


def test_futures():
    """Mailbox awaits futures before passing them to readers."""
    # Timeouts are longer for this example,
    # since they involve creating subprocesses.
    exc = concurrent.futures.ProcessPoolExecutor()
    futures = [exc.submit(_task, i) for i in range(3)]
    mailbox_tester(futures,
                   expected_result=[0, 1, 2],
                   result_timeout=5 * LONG_TIMEOUT,
                   timeout=5 * LONG_TIMEOUT)
