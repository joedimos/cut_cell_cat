"""Atomic JSON output: failed serialization never destroys an existing result."""
import json
import os
from pathlib import Path
import tempfile


def atomic_json(path, value):
    path = Path(path)
    # Validate the complete payload before opening any output file.
    payload = json.dumps(value, indent=2, allow_nan=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=f'.{path.name}.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
