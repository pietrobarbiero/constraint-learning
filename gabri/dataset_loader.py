import copy
import os
import pickle

import torch
import numpy as np
import os.path
import torchvision
from PIL import Image
from torch._utils import _accumulate
from torch.utils.data import Subset, Dataset
from torchvision.datasets import ImageFolder, VisionDataset
from torchvision.models import resnet50
from shutil import copyfile
from class_1_hots import CLASS_1_HOTS
from utils import find_folder, show_batch

ANIMAL_DATASET      = "animals"
ANIMAL_DATASET_GRAY = "animals_gray"
CIFAR_100           = "cifar100"
CIFAR_100_GRAY      = "cifar100_gray"
PASCAL_PART         = "pascalpart"
PASCAL_PART_GRAY    = "pascalpart_gray"
GRAY                = "_gray"

DATASETS = [ANIMAL_DATASET, CIFAR_100, PASCAL_PART]

NUM_CLASSES = {
	ANIMAL_DATASET: 33,
	CIFAR_100: 120,
	PASCAL_PART: 64
}

MAIN_CLASSES = {
    ANIMAL_DATASET: range(7),
    CIFAR_100: range(100),
    PASCAL_PART: [0, 5, 6, 7, 8, 10, 12, 13, 15, 17, 18, 33, 37, 43, 47, 52, 53, 55, 58, 60],  # classes are alphabetically sorted
}


class SemiSupDataset(ImageFolder):
    """
     Custom Dataset class that extends ImageFolder and provide a supervision mask for semi-supervised learning
     and labels are converted and returned as a one-hot array instead of the class number

    Methods
    -------
     __getitem__() return 3 elements: image, label, and 1 or 0 if it is a supervised sample or not
    """
    def __init__(self, root: str, dataset: str, transform: callable = None, load_in_memory: bool = True,
                 sup_fraction: float = 1.0, partially_labeled: float = None, pos_weights: np.ndarray = None,
                 seed: int = 0, n_samples=None, main_classes=False):
        """
        Parameters:
        ----------
        root (string): Directory with all the images.
        dataset (string): Dataset to use, need to be among those implemented
        transform (callable, optional): Optional transform to be applied on a sample.
        load_in_memory (bool, default True): Load the whole dataset in memory at the beginning of the training
        sup_fraction (float, default 1.0): Number of supervised elements in percentage
        partially_labeled (float, optional): Amount of partially labelling as fraction
        pos_weights (np.ndarrray, optional): Optional class weights
        """

        root = find_folder(root)
        assert dataset in DATASETS, "Unavailable dataset {}. Datasets available: {}".format(dataset, DATASETS)
        super().__init__(root, transform)
        self.sup_fraction       = sup_fraction
        self.partially_labeled  = partially_labeled
        self.dataset_name            = dataset
        if pos_weights is not None:
            self.pos_weight     = pos_weights
        else:
            self.pos_weight = torch.ones(NUM_CLASSES[dataset]) if not main_classes else torch.ones(len(MAIN_CLASSES[dataset]))
            if dataset == ANIMAL_DATASET:
                # Classes not present in the pre-training data of Imagenet
                self.pos_weight[2] = 1.7  # GIRAFFE
                self.pos_weight[4] = 1.2  # PENGUIN
        if self.dataset_name is not PASCAL_PART:
            self._class_1_hots      = CLASS_1_HOTS[self.dataset_name]
        self._convert_targets()
        self.seed               = seed
        self.load_in_memory     = load_in_memory
        self.supervision_mask   = self.get_supervision_mask()
        self.main_classes       = main_classes
        self._loaded_samples: list = []
        if main_classes:
            self.reduce_to_main_classes()
        if n_samples is not None:
            self.reduce_to_n_samples(n_samples)
        if load_in_memory:
            self.load_data()

    def __getitem__(self, idx):
        if self.load_in_memory:
            sample, label = self._loaded_samples[idx], self.targets[idx]
        else:
            sample, label   = super().__getitem__(idx)[0], self.targets[idx]
        supervised = self.supervision_mask[idx]
        return sample, label, supervised

    def get_supervision_mask(self, mask_folder="masks", balanced=True):
        y = np.asarray(self.targets)
        assert len(y.shape) > 1, "Error: expected targets to be a matrix n_sample x n_class, received " + str(y.shape)

        pickle_filename = "mask_of_dataset-" + self.dataset_name + "-" + os.path.basename(self.root) + "_semi_sup-" + str(self.sup_fraction) \
                          + "_partially_labeled-" + str(self.partially_labeled) + "_seed-" + str(self.seed) + ".pickle"
        cur_dir = os.path.abspath(os.curdir)
        os.chdir(self.root)
        try:
            mask_folder = os.path.abspath(find_folder(os.path.join("..", "..", mask_folder)))
        except FileNotFoundError:
            os.mkdir(os.path.join("..", "..", mask_folder))
            mask_folder = os.path.abspath(find_folder(mask_folder))
        os.chdir(mask_folder)
        if os.path.isfile(pickle_filename):
            with open(pickle_filename, 'rb') as f:
                y_mask = pickle.load(f)
            # print("Loaded mask from file")
        else:
            print("Calculating supervision mask...")
            os.chdir(cur_dir)
            if self.pos_weight is None:
                self.pos_weight = np.ones(y.shape[1])
            y_mask = torch.zeros_like(self.targets, dtype=torch.float)
            indexes = []
            if self.dataset_name is PASCAL_PART:
                balanced = False
            if balanced:
                for j, c in enumerate(self.class_to_idx):
                    idx = [i for i in range(len(y)) if np.all(y[i, :] == self._class_1_hots[c])]
                    n = int(self.sup_fraction * len(idx))
                    indexes.extend(idx[:n])
                indexes = np.unique(np.array(indexes))
            else:
                indexes = [i for i in range(y.shape[0])]
                np.random.shuffle(indexes)
                n = int(np.round(self.sup_fraction * y.shape[0]))
                indexes = indexes[:n]

            for i in indexes:
                for k, p in enumerate(self.pos_weight):
                    y_mask[i, k] = p if self.targets[i, k] else 1

            if self.partially_labeled is not None:
                for j in range(y_mask.shape[1]):
                    pos_supervised_mask_j = np.intersect1d(np.argwhere(y_mask[:, j] != 0), np.argwhere(y[:, j] == 1))
                    neg_supervised_mask_j = np.intersect1d(np.argwhere(y_mask[:, j] != 0), np.argwhere(y[:, j] == 0))
                    np.random.shuffle(pos_supervised_mask_j), np.random.shuffle(neg_supervised_mask_j)
                    n_pos_sup_mask_to_zero = int((1-self.partially_labeled) * len(pos_supervised_mask_j))
                    n_neg_sup_mask_to_zero = int((1-self.partially_labeled) * len(neg_supervised_mask_j))
                    idx2_pos_sup_to_zero = pos_supervised_mask_j[:n_pos_sup_mask_to_zero]
                    idx2_neg_sup_to_zero = neg_supervised_mask_j[:n_neg_sup_mask_to_zero]
                    y_mask[idx2_pos_sup_to_zero, j] = 0
                    y_mask[idx2_neg_sup_to_zero, j] = 0

            print("Saving supervision mask file for future runs")
            os.chdir(mask_folder)
            with open(pickle_filename, 'wb') as f:
                pickle.dump(y_mask, f, protocol=4)

        if self.partially_labeled is None:
            sup_fraction = np.sum(y_mask.numpy() != 0) / y.size
            assert np.abs(sup_fraction - self.sup_fraction) < 0.01, \
                "Something went wrong, dataset should have %.2f supervisions, instead it has %.2f" \
                % (self.sup_fraction, sup_fraction)
        else:
            sup_fraction = np.sum(y_mask.numpy() != 0) / y.size
            assert np.abs(sup_fraction - self.sup_fraction * self.partially_labeled) < 0.01, \
                "Something went wrong, dataset should have %.2f supervisions, instead it has %.2f" \
                % (self.sup_fraction * self.partially_labeled, sup_fraction)

        os.chdir(cur_dir)
        return y_mask

    def _convert_targets(self):
        if self.dataset_name is PASCAL_PART:
            cur_dir = os.path.abspath(os.curdir)
            os.chdir(self.root)
            with open("labels.pickle", 'rb') as f:
                one_hot_tensor = pickle.load(f)
            os.chdir("..")
            with open("class_to_idx.pickle", 'rb') as f:
                self.class_to_idx = pickle.load(f)
                self.classes = [c for c in self.class_to_idx.keys()]
            with open("object_part_dictionary.pickle", 'rb') as f:
                self._object_parts_dict = pickle.load(f)
            with open("object_part_inv_dictionary.pickle", "rb") as f:
                self._object_parts_inv_dict = pickle.load(f)
            os.chdir(cur_dir)
        else:
            one_hot_tensor: list = [self._class_1_hots[self.classes[t]] for t in self.targets]
        self.targets = torch.Tensor(one_hot_tensor)

    def load_data(self):
        print("Loading dataset...")
        print("%d/%d Samples loaded in memory" % (0, len(self.imgs)))
        i = 0
        for i in range(len(self.imgs)):
            img_sample = super().__getitem__(i)[0]
            self._loaded_samples.append(img_sample)
            if (i + 1) % 1000 == 0:
                print("%d/%d Samples loaded in memory" % (i + 1, len(self.imgs)))
        if (i + 1) % 1000 != 0:
            print("%d/%d Samples loaded in memory" % (i + 1, len(self.imgs)))
        self._loaded_samples = torch.stack(self._loaded_samples)

    def reduce_to_n_samples(self, n_samples):
        assert n_samples <= self.targets.shape[0], "number of samples required is higher than the number of samples " \
                                                  "available"
        n_idx = np.random.random_integers(0, len(self) - 1, n_samples)
        self.imgs               = [self.imgs[idx] for idx in n_idx]
        self.samples            = [self.samples[idx] for idx in n_idx]
        self.targets            = torch.stack([self.targets[idx] for idx in n_idx])
        self.supervision_mask   = torch.stack([self.supervision_mask[idx] for idx in n_idx])

    def reduce_to_main_classes(self):
        self.targets = self.targets[:, MAIN_CLASSES[self.dataset_name]]
        self.supervision_mask = self.supervision_mask[:, MAIN_CLASSES[self.dataset_name]]


class ExtractFeaturesSemiSupDataset(SemiSupDataset):
    """
    ExtractFeaturesSemiSupDataset is a dataset implementing the SemiSupDataset. Features are preprocessed in load_data
    by the backbone of a Resnet50 network in order to speed up training. In this implementation, data are always loaded
    in memory.
    """

    def __init__(self, root: str, dataset: str, transform: callable = None, load_in_memory: bool = True, sup_fraction: float = 1.0,
                 partially_labeled: float = None, pos_weights: np.ndarray = None, fine_tuning: bool = False,
                 device: str = "cpu", seed: int = 0, n_samples=None, main_classes=False):
        """
        Parameters
        ---------
        fine_tuning: bool
        If set, in features are extracted until the last convolutional layer. Otherwise features are passed through all the
        convolutional blocks of the Resent 50
        """
        assert(load_in_memory == True), "Load in memory needs to be true when using ExtractFeaturesSemiSupDataset"
        self._fine_tuning = fine_tuning
        self.device = device
        super().__init__(root, dataset, transform, load_in_memory=load_in_memory, sup_fraction=sup_fraction, partially_labeled=partially_labeled,
                         pos_weights=pos_weights, seed=seed, n_samples=n_samples, main_classes=main_classes)

    def load_data(self):
        pickle_filename = "extracted_feat_ft.pickle" if self._fine_tuning else "extracted_feat_tl.pickle"
        cur_dir = os.path.abspath(os.curdir)
        os.chdir(self.root)
        self._loaded_samples = []
        if os.path.isfile(pickle_filename):
            print("Loading dataset feature from file")
            with open(pickle_filename, 'rb') as f:
                self._loaded_samples = pickle.load(f)
            print("Dataset feature loaded")
        else:
            print("Extracting dataset feature...")
            os.chdir(cur_dir)
            resnet = resnet50(pretrained=True)
            if self._fine_tuning:
                net = torch.nn.Sequential(*list(resnet.children())[:-3])
            else:
                net = torch.nn.Sequential(*list(resnet.children())[:-1])
            batch_images = []
            net.to(self.device)
            net.eval()
            with torch.no_grad():
                print("%d/%d Samples loaded in memory" % (0, len(self.imgs)))
                for i in range(len(self.imgs)):
                    batch_images += [ImageFolder.__getitem__(self, i)[0].to(self.device)]
                    if (i + 1) % 64 == 0:

                        self._loaded_samples += [net(torch.stack(batch_images)).squeeze().to("cpu")]
                        batch_images = []
                    if (i + 1) % 1000 == 0:
                        print("%d/%d Samples loaded in memory" % (i + 1, len(self.imgs)))
                print("%d/%d Samples loaded in memory" % (i + 1, len(self.imgs)))
                if len(batch_images) > 0:
                    self._loaded_samples += [net(torch.stack(batch_images)).squeeze().to("cpu")]
                self._loaded_samples = torch.cat(self._loaded_samples)
                os.chdir(self.root)
                if self.dataset_name != CIFAR_100:
                    with open(pickle_filename, 'wb') as f:
                        print("Saving extracted features on memory")
                        pickle.dump(self._loaded_samples.numpy(), f, protocol=4)
        os.chdir(cur_dir)


class SingleLabelDataset(Dataset):
    """
    SingleLabelDataset is a very simple dataset that receives x and y as np.array already transformed.
    """
    def __init__(self, x: np.array, y: np.array):
        """
        Parameters
        ---------
        x: np.array
        Samples of the dataset
        y: np.array
        labels of the dataset
        """
        super().__init__()
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y

    def __len__(self):
        return self.x.shape[0]


def create_dataset_splits(dataset: VisionDataset, dest_root: str, splits: list, names: list, balanced=True, ext="JPEG"):
    subsets = get_fixed_splits(dataset, splits, balanced)
    if not os.path.isdir(dest_root):
        os.makedirs(dest_root)
    for subset, name in zip(subsets, names):
        split_folder = os.path.join(dest_root, name)
        if not os.path.exists(split_folder):
            os.makedirs(split_folder)
        if os.path.basename(dest_root) == "pascal_splits":
            subset.dataset.classes = ["images"]
        for c in subset.dataset.classes:
            target_class = os.path.join(split_folder, c)
            if not os.path.exists(target_class):
                os.makedirs(target_class)
        labels = []
        images = []
        subset.indices = sorted(subset.indices)
        for i in subset.indices:
            src, label, _ = subset.dataset.__getitem__(i)
            images.append(src), labels.append(label)
            if len(images) == 4:
                show_batch(images, labels, dataset.classes)
            if len(subset.dataset.classes) > 1:
                if isinstance(label, torch.Tensor):
                    class_folder = subset.dataset.classes[np.argwhere(label.numpy())[0][0]]
                else:
                    class_folder = subset.dataset.classes[label]
            else:
                class_folder = subset.dataset.classes[0]
            filename = "{:06d}.{}".format(i, ext)
            dst = os.path.join(split_folder, class_folder, filename)
            if isinstance(src, torch.Tensor):
                im = Image.fromarray(src.numpy())
            elif isinstance(src, np.ndarray):
                im = Image.fromarray(src)
            else:
                im = src
            im.save(dst)
        labels = torch.stack(labels)
        if os.path.basename(dest_root) == "pascal_splits":
            with open(os.path.join(split_folder, "labels.pickle"), 'wb') as f:
                pickle.dump(labels, f)


def get_fixed_splits(dataset: Dataset, splits: list, balanced=True):
    indices = {i: [] for i in range(len(splits))}
    targets = np.asarray(dataset.targets)
    if balanced:
        classes = np.unique(targets, axis=0)
        for j, c in enumerate(classes):
            class_targets = np.argwhere([np.all(t == c) for t in targets]).squeeze()
            class_lengths = [int(np.round(split*class_targets.size)) for split in splits]
            if np.sum(class_lengths) < class_targets.size:
                class_lengths[-1] += class_targets.size - np.sum(class_lengths)
            elif np.sum(class_lengths) > class_targets.size:
                class_lengths[-1] -= np.sum(class_lengths) - class_targets.size
            if class_targets.size > 1:
                class_indexes = [class_targets[o - l:o] for o, l in zip(_accumulate(class_lengths), class_lengths)]
            else:
                class_indexes = [[int(class_targets)] if i == 0 else [] for i in range(len(splits))]
            for i, class_index in enumerate(class_indexes):
                indices[i].extend(class_index)
            if (j + 1) % 100 == 0:
                print(j+1)
    else:
        indexes = [i for i in range(targets.shape[0])]
        np.random.shuffle(indexes)
        splits = [int(np.round(split * targets.shape[0])) for split in splits]
        indices = [indexes[o - l:o] for o, l in zip(_accumulate(splits), splits)]
        indices = {i: idx for i, idx in enumerate(indices)}
    subsets = [Subset(dataset, index) for index in indices.values()]
    for subset in subsets:
        subset.dataset = copy.deepcopy(subset.dataset)
    return subsets


def rename_data_dataset(root: str):
    dataset = ImageFolder(root)
    j = 0
    for i, (img, _) in enumerate(dataset.samples):
        if len(os.path.basename(img)) > 10:
            path = os.path.dirname(img)
            ext = img.split(".")[-1]
            filename = os.path.join(path, "{:06d}.{}".format(i, ext))
            while os.path.isfile(filename):
                j += 1
                filename = os.path.join(path, "{:06d}.{}".format(i+j, ext))
            copyfile(img, filename)
            os.remove(img)


def calc_dataset_mean_std(dataset: Dataset):

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        num_workers=0,
        shuffle=False
    )

    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, _, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

        print("Number of samples", nb_samples)

    mean /= nb_samples
    std /= nb_samples

    print("Mean", mean)
    print("Std", std)

    return mean, std


if __name__ == "__main__":

    np.random.seed(0)
    transform = torchvision.transforms.Compose([
        # torchvision.transforms.Grayscale(),
        # torchvision.transforms.ToTensor(),
    ])

    dataset = SemiSupDataset("..//data//pascalpart", "pascalpart", transform=transform, load_in_memory=False)

    # images, labels = [], []
    # for i in range(4):
    #     images.append(torch.stack([dataset.__getitem__(i)[0]], dim=0))
    #     labels.append(torch.stack([dataset.__getitem__(i)[1]], dim=0))
    #
    # show_batch(images, labels, dataset.classes)
    #
    # print(len(dataset))

    dest_dataset = "..//data//pascal_splits"
    splits = [0.7, 0.15, 0.15]
    names = ["train", "val", "test"]

    create_dataset_splits(dataset, dest_dataset, splits, names, balanced=False)
    #
    # dest_dataset = "../datasets/animals_gray_train_val_test"
    #
    # rename_data_dataset(dest_dataset)

