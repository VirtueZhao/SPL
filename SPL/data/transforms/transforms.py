import random
from PIL import Image
from torchvision.transforms import InterpolationMode, Resize, RandomHorizontalFlip, ToTensor, Normalize, Compose, CenterCrop

AVAILABLE_TRANSFORMS = [
    "normalize",
    "random_flip",
    "random_translation",
]

INTERPOLATION_MODES = {
    "bilinear": InterpolationMode.BILINEAR
}


class Random2DTranslation:
    """ Resized an Image with (height*1.125, width*1.125), then Perform Random Cropping.

    Args:
        height (int): target image height.
        width (int): target image width.
        p (float, optional): probability that this operation takes place.
            Default is 0.5.
        interpolation (int, optional): desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img.resize((self.width, self.height), self.interpolation)

        new_width = int(round(self.width * 1.125))
        new_height = int(round(self.height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)

        x_max_range = new_width - self.width
        y_max_range = new_height - self.height
        x1 = int(round(random.uniform(0, x_max_range)))
        y1 = int(round(random.uniform(0, y_max_range)))
        cropped_img = resized_img.crop(
            (x1, y1, x1 + self.width, y1 + self.height)
        )

        return cropped_img


def build_transform(cfg, is_train=True, transforms=None):
    """Build Transformation Functions.

    Args:
        cfg (CfgNode): config.
        is_train (bool, optional): for training (True) or test (False).
            Default is True.
        transforms (list, optional): list of strings which will overwrite
            cfg.INPUT.TRANSFORMS if given. Default is None.
    """

    if cfg.INPUT.NO_TRANSFORM:
        print("Note: No Transform is Applied.")
        return None

    if transforms is None:
        transforms = cfg.INPUT.TRANSFORMS

    for transform in transforms:
        assert transform in AVAILABLE_TRANSFORMS

    normalize = Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)

    if is_train:
        return _build_transform_train(cfg, transforms, normalize)
    else:
        return _build_transform_test(cfg, transforms, normalize)


def _build_transform_train(cfg, transforms, normalize):
    interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]

    transform_train = []

    # Ensure the Image Size Matches the Target Size
    resize_conditions = ["random_crop" not in transforms, "random_resized_crop" not in transforms]
    if all(resize_conditions):
        transform_train += [Resize(cfg.INPUT.SIZE, interpolation=interp_mode)]

    if "random_translation" in transforms:
        transform_train += [Random2DTranslation(cfg.INPUT.SIZE[0], cfg.INPUT.SIZE[1])]

    if "random_flip" in transforms:
        transform_train += [RandomHorizontalFlip()]

    transform_train += [ToTensor()]

    if "normalize" in transforms:
        transform_train += [normalize]

    transform_train = Compose(transform_train)
    # print("Training Data Transforms: {}".format(transform_train))

    return transform_train


def _build_transform_test(cfg, transforms, normalize):
    interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]

    transform_test = []
    transform_test += [Resize(max(cfg.INPUT.SIZE), interpolation=interp_mode)]
    transform_test += [CenterCrop(cfg.INPUT.SIZE)]
    transform_test += [ToTensor()]

    if "normalize" in transforms:
        transform_test += [normalize]

    transform_test = Compose(transform_test)
    # print("Testing Data Transforms: {}".format(transform_test))

    return transform_test
