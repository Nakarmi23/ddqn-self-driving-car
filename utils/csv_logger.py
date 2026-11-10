import csv
import os


class CsvLogger:
    """Append-only CSV writer that writes a header once per file."""

    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(fieldnames)

    def log(self, *values):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(values)
