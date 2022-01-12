import pickle
import tarfile
import random
import numpy

from ..util import get_file, unzip, to_categorical

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"


def cifar(variant='10', channels_last=False, shuffle=True):
    if variant == '10':
        data = load_cifar10()
    elif variant == '100':
        data = load_cifar100(coarse=False)
    elif variant == '100-coarse':
        data = load_cifar100(coarse=True)
    else:
        raise ValueError("Variant must be one of: '10', '100', 100-coarse")
    X_train, y_train, X_test, y_test = data
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train /= 255.0
    X_test /= 255.0
    if shuffle:
        train_data = list(zip(X_train, y_train))
        random.shuffle(train_data)
        X_train, y_train = unzip(train_data)
    if channels_last:
        X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
        X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
    else:
        X_train = X_train.reshape(X_train.shape[0], 3, 32, 32)
        X_test = X_test.reshape(X_test.shape[0], 3, 32, 32)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return (X_train, y_train), (X_test, y_test)


def load_cifar10(path='cifar-10-python.tar.gz'):
    path = get_file(path, origin=CIFAR10_URL)
    train_images = []
    train_labels = []
    with tarfile.open(path) as cifarf:
        for name in cifarf.getnames():
            # data is stored in batches
            if 'data_batch' in name:
                decompressed = cifarf.extractfile(name)
                data = pickle.load(decompressed, encoding='bytes')
                train_images.append(data[b'data'])
                train_labels += data[b'labels']
            elif 'test_batch' in name:
                decompressed = cifarf.extractfile(name)
                data = pickle.load(decompressed, encoding='bytes')
                test_images = data[b'data']
                test_labels = data[b'labels']
    train_images = numpy.vstack(train_images)
    train_labels = numpy.asarray(train_labels)
    test_labels = numpy.asarray(test_labels)
    return train_images, train_labels, test_images, test_labels


def load_cifar100(path='cifar-100-python.tar.gz', coarse=False):
    path = get_file(path, origin=CIFAR10_URL)
    with tarfile.open(path) as cifarf:
        train_decomp = cifarf.extractfile('cifar-100-python/train')
        test_decomp = cifarf.extractfile('cifar-100-python/test')
        train_data = pickle.load(train_decomp, encoding='bytes')
        test_data = pickle.load(test_decomp, encoding='bytes')
    train_images = train_data[b'data']
    test_images = test_data[b'data']
    if coarse:
        train_labels = train_data[b'coarse_labels']
        test_labels = test_data[b'coarse_labels']
    else:
        train_labels = train_data[b'fine_labels']
        test_labels = test_data[b'fine_labels']
    train_labels = numpy.asarray(train_labels)
    test_labels = numpy.asarray(test_labels)
    return train_images, train_labels, test_images, test_labels
