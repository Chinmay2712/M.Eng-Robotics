# Exercise 3: Why does `publish()` iterate over `list(self._callbacks)` instead of `self._callbacks`?

In `topic.py`, the `publish()` method contains the following code:

```python
for callback in list(self._callbacks):
    callback(msg)
```

The publish() method iterates over list(self._callbacks) to create a snapshot of the subscribers before notifying them. This prevents changes to the original subscriber list during iteration from causing callbacks to be skipped or processed unexpectedly. Subscribers can safely subscribe or unsubscribe while a message is being published. Changes to the subscriber list will affect subsequent publications rather than the current iteration.

Therefore, using `list(self._callbacks)` makes the notification process safer when subscribers are added or removed during message publication.