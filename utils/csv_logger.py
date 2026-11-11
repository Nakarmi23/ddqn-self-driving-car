import csv
import os


class CsvLogger:
    """Append-only CSV writer that writes a header once per file.

    The file stays open between rows. Rows are flushed to disk every
    `flush_every` rows, on flush() and on close(), so a high-frequency log
    (one row per training step) doesn't pay for an open/close per row.
    """

    def __init__(self, path, fieldnames, flush_every=1):
        self.path = path
        self.fieldnames = fieldnames
        self.flush_every = flush_every
        self._pending = 0
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_header = not os.path.exists(path) or os.path.getsize(path) == 0
        self._file = open(path, "a", newline="")
        self._writer = csv.writer(self._file)
        if write_header:
            self._writer.writerow(fieldnames)
            self._file.flush()

    def log(self, *values):
        self._writer.writerow(values)
        self._pending += 1
        if self._pending >= self.flush_every:
            self.flush()

    def flush(self):
        if not self._file.closed:
            self._file.flush()
        self._pending = 0

    def close(self):
        if not self._file.closed:
            self._file.close()
