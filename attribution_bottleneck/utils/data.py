from math import ceil
from mlproject.data import *
import h5py
import io
from os import path
import yaml
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

class DataProvider:
    """
    wrapper class for a dataset factory to retrieve data from
    """
    def __init__(self, data_fac, device, label_dict=None):
        self.data_fac = data_fac
        self.device = device
        self.label_dict = label_dict

    def gen_samples(self, num, device=None, test_set=False) -> list:
        device = device if device else self.device
        loader = self.data_fac.test_loader() if test_set else self.data_fac.train_loader()
        yielded = 0
        for i, (data, labels) in enumerate(loader.__iter__()):
            done = False
            for s in range(data[0].shape[0]):
                yielded += 1
                if yielded <= num:
                    yield (data[s].unsqueeze(0).to(device), labels[s].unsqueeze(0).to(device))
                else:
                    done = True
                    break
            if done:
                break

    def gen_batches(self, num, device=None, test_set=False):
        device = device if device else self.device
        loader = self.data_fac.train_loader() if not test_set else self.data_fac.test_loader()
        for i, (data, labels) in enumerate(loader.__iter__()):
            if i > num:
                break
            yield (data.to(device), labels.to(device))

    def get_label(self, num):
        """ if a label dictionary was provided: read it, otherwise just return plain number """
        return str(num) if self.label_dict is None else self.label_dict[num]

class DataSubsetIterator:
    def __init__(self, loader):
        pass

    def __next__(self):
        pass

class TorchZooImageNetFolderDataProvider(DataProvider):
    """
    seed 1, get 100 samples:
        1=boar
        2=monkey (382)
    """
    def __init__(self, config, transform=None):

        config = self.apply_defaults(config)

        # prepare arguments for ImageFolderDataset
        if transform is None:
            transform = Compose([
                Resize(256),
                CenterCrop((224, 224)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.transform = transform

        # maybe seed
        seed = config.get("seed", None)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(config["seed"])

        # maybe load label dict
        label_dict = None
        dict_file = config.get("imagenet_dict_file", None)
        if dict_file is not None:
            with open(dict_file, encoding='utf-8') as handle:
                label_dict = yaml.load(handle.read())

        device = config["device"]
        data_fac = ImageFolderDatasetFactory(
            batch_size=config["batch_size"],
            num_workers=10,
            data_train_dir=config.get("imagenet_train"),
            data_test_dir=config.get("imagenet_test"),
            train_transform=self.transform,
            test_transform=self.transform)

        super().__init__(data_fac, device, label_dict)

    def image_to_batch(self, path, label:int):
        sample = Image.open(path)
        sample = sample.convert('RGB')
        sample = (self.transform(sample).unsqueeze(0).to(self.device), torch.tensor(label).to(self.device))
        return sample

    @staticmethod
    def apply_defaults(overrides=None):
        return {**{
            'imagenet_dict_file': '../../data/imagenet_full/dict.txt',
            'imagenet_train': '/mnt/ssd/data/imagenet/imagenet-raw/train',
            'imagenet_test': '/mnt/ssd/data/imagenet/imagenet-raw/validation',
        }, **(overrides if overrides is not None else {})}

# dataset factories

class ImageNetDataset:
    def __init__(self, hdf5_filename, train, transform=None):
        self.hdf5_filename = hdf5_filename
        self.train = train
        self.dataset_name = 'train' if train else 'validation'
        self.transform = transform
        self.open = False
        self.h5 = None
        self.h5_images = None
        self.h5_targets = None

        with h5py.File(hdf5_filename, 'r') as tmp_h5:
            h5_targets = tmp_h5[self.dataset_name + '/targets']
            self.length = len(h5_targets)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if not self.open:
            self.h5 = h5py.File(self.hdf5_filename, 'r', swmr=True)
            self.h5_images = self.h5[self.dataset_name + '/images']
            self.h5_targets = self.h5[self.dataset_name + '/targets']
            self.open = True
        target = self.h5_targets[idx]
        jpg_bytes = self.h5_images[idx].tobytes()
        pil_image = Image.open(io.BytesIO(jpg_bytes))
        if self.transform is not None:
            img = self.transform(pil_image)
        else:
            img = pil_image
        return img, int(target)

class ImageFolderDatasetFactory(TorchvisionDatasetFactory):
    def __init__(self, batch_size,
                 data_train_dir,
                 data_test_dir,
                 train_transform=None,
                 test_transform=None,
                 loader=None,
                 num_workers=0):
        assert path.exists(data_train_dir), "train does not exist: {}".format(data_train_dir)
        assert path.exists(data_test_dir), "test does not exist: {}".format(data_test_dir)
        loader = loader if loader is not None else torchvision.datasets.folder.default_loader
        resource_train = torchvision.datasets.ImageFolder(data_train_dir, transform=train_transform, loader=loader)
        resource_test = torchvision.datasets.ImageFolder(data_test_dir, transform=test_transform, loader=loader)
        super().__init__(
            resource_train, resource_test,
            data_loader_kwargs={
                'num_workers': num_workers,
                'batch_size': batch_size,
            },
            data_loader_train_kwargs={'shuffle': True}
        )

class ImageNetHDF5DatasetFactory(TorchvisionDatasetFactory):
    def __init__(self, batch_size,
                 train_transform=None,
                 test_transform=None,
                 data_file=None,
                 dict_file=None,
                 num_workers=0):
        resource_train = ImageNetDataset(data_file, transform=train_transform, train=True)
        resource_test = ImageNetDataset(data_file, transform=test_transform, train=False)
        with open(dict_file, encoding='utf-8') as handle:
            self.dict = yaml.load(handle.read())
        super().__init__(
            resource_train, resource_test,
            data_loader_kwargs={
                'num_workers': num_workers,
                'batch_size': batch_size,
                'shuffle': True,
            },
            data_loader_train_kwargs={}
        )

    def get_label(self, num):
        return self.dict[num]

class TorchZooHDF5ImageNetDatasetFactory(ImageNetHDF5DatasetFactory):
    def __init__(self, config):
        train_transform = Compose([
            Resize(256),
            CenterCrop((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        test_transform = Compose([
            Resize(256),
            CenterCrop((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        seed = config.get("seed", None)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed()
        super().__init__(
            batch_size=config["batch_size"],
            num_workers=10,
            data_file=config.get("data_file"),
            dict_file=config.get("dict_file"),
            train_transform=train_transform,
            test_transform=test_transform)


class ClutMNISTNormalizedDataProvider(DataProvider):
    def __init__(self, config):

        # maybe seed
        seed = config.get("seed", None)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed()

        config = self.apply_defaults(config)
        device = config["device"]
        data_resize = config.get("data_resize", (96, 96))
        train_transform = Compose([
            Resize(data_resize),
            ToTensor(),
            Normalize((0.5,), (0.5,))])
        test_transform = Compose([
            Resize(data_resize),
            ToTensor(),
            Normalize((0.5,), (0.5,))])
        data_fac = ClutteredMNISTDatasetFactory(
            batch_size=config.get("batch_size", 32),
            shape=(96, 96),
            num_workers=10,
            use_filesys=config.get("data_use_filesys", True),
            n_clutters=config.get("clut_mnist_clutters"),
            data_dir=config.get("clut_mnist_data_dir", None),
            train_transform=train_transform,
            test_transform=test_transform,
            n_samples_train=config.get("clut_mnist_train_samples"),
            n_samples_test=int(0.15 * config.get("clut_mnist_train_samples")))
        super().__init__(data_fac, device)

    @staticmethod
    def apply_defaults(overrides=None):
        return {**{
            'clut_mnist_train_samples':60021,
            'clut_mnist_clutters': 21,
            'clut_mnist_data_dir': '../../data/cluttered_mnist_60021',
        }, **(overrides if overrides is not None else {})}


class MNISTNormalizedDatasetFactory(MNISTDatasetFactory):
    def __init__(self, config):
        data_resize = config.get("data_resize", (32, 32))
        train_transform = Compose([
            Resize(data_resize),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_transform = Compose([
            Resize(data_resize),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        super().__init__(
            batch_size=config["batch_size"],
            train_transform=train_transform,
            test_transform=test_transform)
