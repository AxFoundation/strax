"""Single-threaded message bus / mailbox-system replacement code"""
import typing as ty


class Spy:
    """Template for spies; a spy that does nothing."""

    def receive(self, msg):
        """Called when a new message is produced"""
        pass

    def close(self):
        """Called when the topic is exhausted"""
        pass

    def kill(self, reason):
        """Called when closing the spy prematurely, e.g. during
        exception handling."""
        self.close()


class PostOffice:
    """A single-threaded message bus that uses iterators.

    This allows you to register producers and create readers of messages
    of different topics. You can also register 'spies', which will get each
    message just after it has been produced.

    Notes:
      * The readers are iterators (technically generators), and the producers
        should be iterable as well (probably you implement them as generators).
      * Only one producer can be registered for each topic.
      * Producers may produce (topic -> message) dicts and thereby feed multiple
        topics at once.
      * If multiple readers are registered for the same topic, PostOffice
        will save messages not yet read by all readers.
      * If you create a reader and never iterate it to completion, messages
        will be saved until the PostOffice is garbage collected.
      * We only call .close() on a spy when the topic is exhausted. To close
        all spies prematurely (e.g. to handle an exception), call
        .kill_spies().
    """

    def __init__(self):
        # Set of topics that have been exhausted (no more messages will come)
        self._exhausted_topics: ty.Set[str] = set()
        # Set of topics that are multi output
        # (i.e. the producer makes topic -> message dicts)
        self._multi_output_topics: ty.Set[str] = set()

        # Per-topic state
        # (Could refactor everything into a Topic class that PostOffice
        #  would be a shell for... not sure that is actually nicer.)

        # Dict: topic -> list with (msg_number, msg)
        self._saved_mail: ty.Dict[str, ty.Tuple[int, ty.Any]] = dict()
        # Dict: topic -> list of spies
        self._spies: ty.Dict[str, ty.List[Spy]] = dict()
        # Dict: topic-> iterator that produces messages
        self._producers: ty.Dict[str, ty.Iterable] = dict()
        # Dict: topic -> last message produced
        self._last_msg_produced = dict()
        # Dict: topic -> reader_name -> last message number recieved
        self._last_msg_read: ty.Dict[str, ty.Dict[str, int]] = dict()
        # Dict: topic -> list of readers that are done
        self._readers_done: ty.Dict[str, ty.List[str]] = dict()

    def state(self):
        """Return state representation as a multi-line string,
        suitable for printing or logging"""
        result = []
        for topic in self._saved_mail:
            result.append(f"Topic {topic}:")
            result.append(f'  Saved mail: {self._saved_mail[topic]}')
            result.append(f'  Last produced: {self._last_msg_produced[topic]}')
            result.append(f'  Readers recieved: {self._last_msg_read[topic]}')
            result.append(f'  Readers done: {self._readers_done[topic]}')
            result.append('')
        return "\n".join(result)

    def register_producer(
            self,
            iterator: ty.Iterator[ty.Any],
            topic: ty.Union[str, ty.Tuple[str]]):
        """Register iterator as the source of messages for topic.

        If topic is a tuple of strings, the iterator should produce
        (topic -> message) dicts, with every registered topic in the dict."""
        if isinstance(topic, tuple):
            if len(topic) == 1:
                topic = topic[0]
            else:
                self._multi_output_topics.add(topic)
                for sub_topic in topic:
                    assert isinstance(sub_topic, str)
                    self.register_producer(sub_topic, iterator)
                    return

        if topic in self._producers:
            raise RuntimeError(f"{topic} already has a producer")
        self._register_topic(topic)
        self._producers[topic] = iterator

    def _register_topic(self, topic: str):
        if topic in self._saved_mail:
            return
        assert isinstance(topic, str)
        self._saved_mail[topic] = []
        self._spies[topic] = []
        self._last_msg_read[topic] = dict()
        self._last_msg_produced[topic] = -1
        self._readers_done[topic] = []

    def register_spy(self, spy: Spy, topic: str):
        """Register spy to recieve all messages on topic.
        spy.recieve(msg) will be called for each message, and
        spy.close() when the topic is exhausted.
        """
        self._register_topic(topic)
        self._spies[topic].append(spy)

    def get_iter(self, topic: str, reader: str):
        """Return iterator over messages with topic, for a named reader
        (usually readers are named after the messages they produce)
        """
        self._register_topic(topic)
        # Register subscriber
        self._last_msg_read[topic][reader] = -1
        # Return generator
        return self._read(topic, reader)

    def kill_spies(self, reason=None):
        """Close all spies immediately, e.g. during exception handling.
        Reason is passed to spy.kill.
        """
        for spies in self._spies.values():
            for spy in spies:
                spy.kill(reason)

    def _read(self, topic, reader):
        """Actual generator producing messages for reader on topic"""
        msg_number = 0
        while self._message_may_come(topic, msg_number):
            # Try to get this message from the cache
            for _msg_i, result in self._saved_mail[topic]:
                if _msg_i == msg_number:
                    break
            else:
                try:
                    # Relinquish control to the producer
                    result = self._fetch_new(topic)
                except StopIteration:
                    # Message actually won't come, exit the while loop.
                    # (The while condition wasn't triggered because
                    #  the producer only just realized the topic is exhausted)
                    break
            # Note receipt before yielding, so we can clear unnecessary
            # messages from our storage (if possible) before we lose control.
            self._ack_reader_recieved(reader, topic, msg_number)
            yield result
            # Look for the next message
            msg_number += 1
        # We get here if the topic is exhausted & we have read all messages
        # For debugging purposes, note that the reader is done before
        # returning / raising StopIteration in the caller.
        self._readers_done[topic].append(reader)

    def _message_may_come(self, topic, msg_number):
        """Return True if topic is guaranteed to never produce msg_number"""
        return not (
            'topic' in self._exhausted_topics
            and msg_number > self._last_msg_read[topic])

    def _fetch_new(self, topic):
        """Fetch a new message from the producer of topic.

        Raises StopIteration if the topic is exhausted so a new message
        will never come.
        """
        if topic not in self._producers:
            raise RuntimeError(f'No producer registered for {topic}')
        try:
            msg = next(self._producers[topic])
        except StopIteration:
            self._ack_topic_exhausted(topic)
            # reraise to end the generator in _read
            raise StopIteration

        if topic not in self._multi_output_topics:
            # Simple message, just ack and return to caller
            self._ack_msg_produced(msg, topic)
            return msg

        # msg is a dict with messages for different topics
        assert isinstance(msg, dict)
        for sub_msg_topic, sub_msg in msg.items():
            if sub_msg_topic == topic:
                # This is what our caller wants
                desired_sub_msg = sub_msg
            self._ack_msg_produced(sub_msg, sub_msg_topic)
        return desired_sub_msg

    def _ack_msg_produced(self, msg, topic):
        """Note that msg of topic has been produced"""
        assert topic not in self._exhausted_topics

        self._last_msg_produced[topic] += 1

        if len(self._last_msg_read.get(topic, [])):
            # Someone is interested in this topic, so save the message.
            # (If there is only one reader, _ack_reader_recieved will clean
            # up this message before we yield control to that reader.)
            self._saved_mail[topic].append(
                (self._last_msg_produced[topic], msg))

        # Deliver the message to the spies (savers/monitors)
        for spy in self._spies[topic]:
            spy.receive(msg)

    def _ack_reader_recieved(self, reader, topic, msg_number):
        """Acknowledge reader got msg_number of topic"""
        # Record receipt
        assert self._last_msg_read[topic][reader] == msg_number - 1
        self._last_msg_read[topic][reader] = msg_number
        # Keep only messages someone has not yet recieved
        everyone_got = max(self._last_msg_read[topic].values())
        self._saved_mail[topic] = [
            (msg_number, msg) for msg_number, msg in self._saved_mail[topic]
            if msg_number >= everyone_got]

    def _ack_topic_exhausted(self, topic):
        """Take note that topic is exhausted, no new messages will come"""
        for spy in self._spies[topic]:
            spy.close()
        self._exhausted_topics.add(topic)
