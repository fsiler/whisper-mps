"""Discover .aax files and queue the smallest files first."""

from __future__ import annotations

import os
import stat
from typing import BinaryIO


def sorted_aax_paths(root: str) -> list[str]:
    """Return regular .aax files ordered by size, then case-insensitive path."""
    if not os.path.isdir(root):
        raise NotADirectoryError(root)

    files = []

    def raise_walk_error(error: OSError) -> None:
        raise error

    for directory, _subdirs, names in os.walk(root, onerror=raise_walk_error):
        for name in names:
            if not name.casefold().endswith(".aax"):
                continue
            path = os.path.join(directory, name)
            info = os.stat(path, follow_symlinks=False)
            if stat.S_ISREG(info.st_mode):
                files.append((info.st_size, path.casefold(), path))

    files.sort()
    return [path for _size, _folded_path, path in files]


def write_sorted_aax_paths(root: str, output: BinaryIO) -> int:
    """Write paths as NUL-delimited bytes, safe for spaces and newlines."""
    paths = sorted_aax_paths(root)
    for path in paths:
        output.write(os.fsencode(path))
        output.write(b"\0")
    return len(paths)


def main(argv: list[str] | None = None) -> int:
    """Write NUL-delimited .aax paths ordered by ascending file size."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("drive_root")
    args = parser.parse_args(argv)
    try:
        write_sorted_aax_paths(args.drive_root, sys.stdout.buffer)
    except OSError as exc:
        print(f"Could not scan Google Drive: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
