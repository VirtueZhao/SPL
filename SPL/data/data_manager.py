import torch
from PIL import Image
from tabulate import tabulate
from .transforms import build_transform
from .datasets.build_dataset import build_dataset
from torch.utils.data import Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torchvision.transforms import InterpolationMode, Resize, ToTensor,  Normalize, Compose

INTERPOLATION_MODES = {
    "bilinear": InterpolationMode.BILINEAR
}


def build_data_loader(
    cfg,
    sampler_type="SequentialSampler",
    data_source=None,
    batch_size=64,
    transform=None,
    is_train=True,
    dataset_wrapper=None
):
    if sampler_type == "RandomSampler":
        sampler = RandomSampler(data_source)
    elif sampler_type == "SequentialSampler":
        sampler = SequentialSampler(data_source)
    else:
        raise ValueError("Unknown Sampler Type :{}".format(sampler_type))

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset_wrapper(cfg, data_source, transform),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=is_train,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA)
    )
    assert len(data_loader) > 0

    return data_loader


class DataManager:

    def __init__(self, cfg, custom_transform_train=None, custom_transform_test=None, dataset_wrapper=None):
        # Load Dataset
        dataset = build_dataset(cfg)

        # Build Transform
        if custom_transform_train is None:
            transform_train = build_transform(cfg, is_train=True)
        else:
            transform_train = custom_transform_train

        if custom_transform_test is None:
            transform_test = build_transform(cfg, is_train=False)
        else:
            transform_test = custom_transform_test

        train_data_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN.SAMPLER,
            data_source=dataset.train_data,
            batch_size=cfg.DATALOADER.TRAIN.BATCH_SIZE,
            transform=transform_train,
            is_train=True,
            dataset_wrapper=dataset_wrapper
        )

        test_data_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test_data,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            transform=transform_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper
        )

        self._num_classes = dataset.num_classes
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._class_label_to_class_name_mapping = dataset.class_label_to_class_name_mapping

        self.dataset = dataset
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader

        self.show_dataset_summary(cfg)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_source_domains(self):
        return self._num_source_domains

    @property
    def class_label_to_class_name_mapping(self):
        return self._class_label_to_class_name_mapping

    def show_dataset_summary(self, cfg):
        dataset_table = [["Dataset", cfg.DATASET.NAME]]
        domain_names = cfg.DATASET.SOURCE_DOMAINS + cfg.DATASET.TARGET_DOMAIN
        domain_names.sort()
        for domain_name in domain_names:
            dataset_table.append([domain_name, f"{self.dataset.domain_info[domain_name]:,}"])

        dataset_table.extend([
            ["Source Domains", cfg.DATASET.SOURCE_DOMAINS],
            ["Target Domain", cfg.DATASET.TARGET_DOMAIN],
            ["# Classes", f"{self.num_classes:,}"],
            ["# Train Data", f"{len(self.dataset.train_data):,}"],
            ["# Test Data", f"{len(self.dataset.test_data):,}"]
        ])

        print(tabulate(dataset_table))


class DatasetWrapper(Dataset):

    def __init__(self, cfg, data_source, transform=None):
        self.cfg = cfg
        self.data_source = data_source
        self.transform = transform
        self.return_original_img = cfg.DATALOADER.RETURN_ORIGINAL_IMG

        interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
        to_tensor = []
        to_tensor += [Resize(cfg.INPUT.SIZE, interpolation=interp_mode)]
        to_tensor += [ToTensor()]
        if "normalize" in cfg.INPUT.TRANSFORMS:
            to_tensor += [Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)]
        self.to_tensor = Compose(to_tensor)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, index):
        datum = self.data_source[index]

        output = {
            "img_path": datum.img_path,
            "domain_label": datum.domain_label,
            "class_label": datum.class_label,
            "index": index
        }

        original_img = Image.open(datum.img_path).convert("RGB")
        output["img"] = self.transform(original_img)

        if self.return_original_img:
            output["original_img"] = original_img

        return output

