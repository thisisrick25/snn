"""Image preprocessing: tensor conversion, normalization, and flattening.

MNIST flattens to [784] (1x28x28); CIFAR-10 flattens to [3072] (3x32x32). Both
produce a flat float tensor so the same MLP-LIF model consumes either dataset.
"""

from torchvision import transforms

# MNIST channel mean/std (standard values).
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

# CIFAR-10 per-channel mean/std (standard values).
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def mnist_transform():
    """Return the MNIST transform: flat float tensor of shape [784] per image."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
            transforms.Lambda(lambda x: x.view(-1)),  # [1,28,28] -> [784]
        ]
    )


def cifar10_transform():
    """Return the CIFAR-10 transform: flat float tensor of shape [3072] per image."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            transforms.Lambda(lambda x: x.view(-1)),  # [3,32,32] -> [3072]
        ]
    )


def cifar10_conv_transform():
    """Return the CIFAR-10 transform for the conv frontend: image tensor [3,32,32].

    Unlike ``cifar10_transform`` this keeps the spatial layout (no flatten) so a
    spiking convolutional frontend can consume the raw image.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
