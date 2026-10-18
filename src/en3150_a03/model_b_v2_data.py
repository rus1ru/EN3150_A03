"""Train-only normalization and orientation-preserving augmentation for Model B v2."""
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from en3150_a03.data import EuroSATDataset


def seed_worker(worker_id):
    random.seed(torch.initial_seed() % (2**32))


class QuarterTurn:
    def __call__(self, image):
        return TF.rotate(image, 90 * int(torch.randint(4, ()).item()))


def training_statistics(root, splits, batch_size=256):
    dataset = EuroSATDataset(root, Path(splits)/"train.csv", transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    sums, squares, pixels = torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64), 0
    for images, _ in loader:
        images = images.to(torch.float64)
        sums += images.sum((0, 2, 3))
        squares += images.square().sum((0, 2, 3))
        pixels += images.shape[0] * images.shape[2] * images.shape[3]
    mean = sums / pixels
    std = (squares/pixels - mean.square()).clamp_min(1e-12).sqrt()
    return {"mean": mean.tolist(), "std": std.tolist(), "source": "train.csv only", "pixels": pixels}


def make_transforms(statistics, augment=True):
    tail = [transforms.ToTensor(), transforms.Normalize(statistics["mean"], statistics["std"])]
    training = [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), QuarterTurn(),
                transforms.RandomCrop(64, padding=4, padding_mode="reflect"),
                transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.08)] if augment else []
    return transforms.Compose(training + tail), transforms.Compose(tail)


def make_loaders(root, splits, statistics, batch_size=128, workers=2, seed=42, augment=True, include_test=False):
    training, evaluation = make_transforms(statistics, augment)
    names = ("train", "val", "test") if include_test else ("train", "val")
    result = {}
    for name in names:
        dataset = EuroSATDataset(root, Path(splits)/f"{name}.csv", training if name == "train" else evaluation)
        result[name] = DataLoader(dataset, batch_size=batch_size, shuffle=name == "train",
            num_workers=workers, pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(seed), persistent_workers=False)
    return result
