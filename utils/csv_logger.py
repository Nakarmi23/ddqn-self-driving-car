import csv
import io
import os


class CsvLogger:
    """Append-only CSV writer that writes a header once per file.

    The file stays open between rows so a high-frequency log (one row per
    training step) doesn't pay for an open/close per row. Rows are buffered
    in memory and written to the file as whole rows every `flush_every`
    rows, on flush() and on close(). A hard kill loses at most the buffered
    rows; if it still cut a row in half, the next run starts a fresh line
    instead of appending onto the partial one.
    """

    def __init__(self, path, fieldnames, flush_every=1):
        self.path = path
        self.fieldnames = fieldnames
        self.flush_every = flush_every
        self._pending = 0
        self._buffer = io.StringIO()
        self._writer = csv.writer(self._buffer)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            needs_header = False
            needs_newline = not self._ends_with_newline(path)
        else:
            needs_header, needs_newline = True, False
        self._file = open(path, "a", newline="")
        if needs_newline:
            # A previous run was killed mid-row; terminate that line so the
            # next row doesn't get glued onto it.
            self._file.write("\r\n")
        if needs_header:
            self._writer.writerow(fieldnames)
        self.flush()

    @staticmethod
    def _ends_with_newline(path):
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            return f.read(1) == b"\n"

    def log(self, *values):
        self._writer.writerow(values)
        self._pending += 1
        if self._pending >= self.flush_every:
            self.flush()

    def flush(self):
        if self._file.closed:
            return
        data = self._buffer.getvalue()
        if data:
            self._file.write(data)
            self._buffer.seek(0)
            self._buffer.truncate()
        self._file.flush()
        self._pending = 0

    def close(self):
        if not self._file.closed:
            self.flush()
            self._file.close()
