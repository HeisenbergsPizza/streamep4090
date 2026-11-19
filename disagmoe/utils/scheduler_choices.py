"""Host timestamps and batch metadata for successful scheduler dequeues."""

import csv
from pathlib import Path
from struct import Struct
from tempfile import NamedTemporaryFile


# One packed entry per nonempty dequeue, independent of hidden size or token count.
_ENTRY = Struct("<QBII")
_STAGES = ("attention", "expert", "finalize")


class SchedulerChoices:
    def __init__(self, num_layers):
        self.num_layers = num_layers
        self._entries = bytearray()

    def record(self, time_ns, metadata):
        layer = metadata.layer_id
        if metadata.is_expert():
            stage = 1
        elif layer == self.num_layers:
            stage = 2
        else:
            stage = 0
        self._entries.extend(_ENTRY.pack(time_ns, stage, layer, metadata.num_tokens()))

    def dump(self, output_dir, device_id, start_ns, end_ns):
        """Publish this worker's choices in the driver clock window [start_ns, end_ns)."""
        if end_ns <= start_ns:
            raise ValueError("The scheduler choice window must have positive duration")
        target = Path(output_dir) / f"choices-gpu-{device_id}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        # The driver can retry a failed dump without reading a partial CSV.
        handle = NamedTemporaryFile("w", encoding="utf-8", newline="", dir=target.parent,
                                    prefix=f".{target.name}.", delete=False)
        temp = Path(handle.name)
        try:
            with handle:
                writer = csv.writer(handle)
                writer.writerow(("time_ns", "stage", "layer", "batch_size"))
                # The inference thread can keep polling while the actor exports.
                # Iterate a snapshot so it remains free to append new entries.
                for stamp, stage, layer, size in _ENTRY.iter_unpack(bytes(self._entries)):
                    if stamp >= end_ns:
                        break
                    if stamp >= start_ns:
                        writer.writerow((stamp, _STAGES[stage], layer, size))
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)
