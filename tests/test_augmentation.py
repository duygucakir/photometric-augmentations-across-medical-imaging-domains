import unittest

import torch
from torchvision.transforms import Normalize

from augmentation.perturbation import ChannelPerturbation
from data.data_loader import get_transforms


class ChannelPerturbationTests(unittest.TestCase):
    def test_perturbation_clips_and_changes_only_selected_channels(self):
        torch.manual_seed(42)
        image = torch.full((3, 16, 16), 0.99)

        result = ChannelPerturbation(["R"], sigma=102.0)(image)

        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)
        self.assertFalse(torch.equal(result[0], image[0]))
        self.assertTrue(torch.equal(result[1], image[1]))
        self.assertTrue(torch.equal(result[2], image[2]))

    def test_perturbation_precedes_imagenet_normalization(self):
        perturbation = ChannelPerturbation(["B"], sigma=12.75)
        pipeline = get_transforms(
            img_size=64,
            is_train=True,
            perturbation=perturbation,
        )

        perturbation_index = pipeline.transforms.index(perturbation)
        normalization_index = next(
            index
            for index, transform in enumerate(pipeline.transforms)
            if isinstance(transform, Normalize)
        )

        self.assertLess(perturbation_index, normalization_index)


if __name__ == "__main__":
    unittest.main()
