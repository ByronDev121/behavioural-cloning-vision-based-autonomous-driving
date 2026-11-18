from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.io import savemat

from data import load_dataset, split_samples


class LoadDatasetTests(unittest.TestCase):
    def test_matlab_images_are_reordered_and_scaled(self):
        images = np.arange(2 * 3 * 1 * 4, dtype=np.uint8).reshape(2, 3, 1, 4)
        targets = np.arange(8, dtype=np.float64).reshape(4, 2)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sample.mat"
            savemat(path, {"X": images, "y": targets})

            loaded_images, loaded_targets = load_dataset(path)

        self.assertEqual(loaded_images.shape, (4, 1, 2, 3))
        np.testing.assert_allclose(loaded_images[0, 0], images[:, :, 0, 0] / 255.0)
        np.testing.assert_array_equal(loaded_targets, targets.astype(np.float32))

    def test_rejects_mismatched_target_count(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sample.mat"
            savemat(path, {"X": np.zeros((2, 2, 1, 3)), "y": np.zeros((2, 2))})

            with self.assertRaisesRegex(ValueError, "count mismatch"):
                load_dataset(path)


class SplitSamplesTests(unittest.TestCase):
    def test_validation_uses_tail_of_ordered_data(self):
        images = np.arange(10).reshape(10, 1, 1, 1)
        targets = np.repeat(np.arange(10)[:, None], 2, axis=1)

        train_x, train_y, val_x, val_y = split_samples(images, targets, 0.2)

        np.testing.assert_array_equal(train_x[:, 0, 0, 0], np.arange(8))
        np.testing.assert_array_equal(train_y[:, 0], np.arange(8))
        np.testing.assert_array_equal(val_x[:, 0, 0, 0], np.arange(8, 10))
        np.testing.assert_array_equal(val_y[:, 0], np.arange(8, 10))


if __name__ == "__main__":
    unittest.main()
