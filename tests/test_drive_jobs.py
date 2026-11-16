import io
import os
import tempfile
import unittest

from whisper_mps.drive_jobs import sorted_aax_paths, write_sorted_aax_paths


class DriveJobOrderingTests(unittest.TestCase):
    def test_discovers_regular_aax_files_and_sorts_smallest_first(self):
        with tempfile.TemporaryDirectory() as root:
            nested = os.path.join(root, "nested folder")
            os.mkdir(nested)
            largest = os.path.join(root, "largest.aax")
            tie_b = os.path.join(nested, "B.AAX")
            tie_a = os.path.join(root, "a.aax")
            smallest = os.path.join(root, "smallest\nfile.aAx")
            for path, contents in (
                (largest, b"12345"),
                (tie_b, b"123"),
                (tie_a, b"abc"),
                (smallest, b"x"),
            ):
                with open(path, "wb") as output:
                    output.write(contents)

            os.mkdir(os.path.join(root, "directory.aax"))
            os.symlink(largest, os.path.join(root, "link.aax"))

            result = sorted_aax_paths(root)

        self.assertEqual(result, [smallest, tie_a, tie_b, largest])

    def test_queue_output_is_nul_delimited_and_preserves_newlines(self):
        with tempfile.TemporaryDirectory() as root:
            paths = [os.path.join(root, name) for name in ("b.aax", "a\nfile.aax")]
            for path, contents in zip(paths, (b"22", b"1")):
                with open(path, "wb") as output:
                    output.write(contents)

            output = io.BytesIO()
            count = write_sorted_aax_paths(root, output)

        self.assertEqual(count, 2)
        self.assertEqual(output.getvalue().split(b"\0"), [os.fsencode(paths[1]), os.fsencode(paths[0]), b""])


if __name__ == "__main__":
    unittest.main()
