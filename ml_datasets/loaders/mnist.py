import random
import gzip
import numpy as np
from srsly import cloudpickle as pickle

from ..util import unzip, to_categorical, get_file
from .._registry import register_loader


MNIST_URL = "https://s3.amazonaws.com/img-datasets/mnist.pkl.gz"

FA_TR_IMG_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz"
FA_TR_LBL_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz"
FA_TE_IMG_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz"
FA_TE_LBL_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz"

KU_TR_IMG_URL = "http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-imgs.npz"
KU_TR_LBL_URL = "http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-labels.npz"
KU_TE_IMG_URL = "http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-imgs.npz"
KU_TE_LBL_URL = "http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-labels.npz"


@register_loader("mnist")
def mnist(variant='mnist', shuffle=True):
    if variant == 'mnist':
        (X_train, y_train), (X_test, y_test) = load_mnist()
    elif variant == 'fashion':
        (X_train, y_train), (X_test, y_test) = load_fashion_mnist()
    elif variant == 'kuzushiji':
        (X_train, y_train), (X_test, y_test) = load_kuzushiji_mnist()
    else:
        raise ValueError("variant must be one of: 'mnist', 'fashion', 'kuzushiji'")
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train /= 255.0
    X_test /= 255.0
    if shuffle:
        train_data = list(zip(X_train, y_train))
        random.shuffle(train_data)
        X_train, y_train = unzip(train_data)
    y_train = to_categorical(y_train, n_classes=10)
    y_test = to_categorical(y_test, n_classes=10)
    return (X_train, y_train), (X_test, y_test)


def load_mnist(path="mnist.pkl.gz"):
    path = get_file(path, origin=MNIST_URL)
    if path.endswith(".gz"):
        f = gzip.open(path, "rb")
    else:
        f = open(path, "rb")
    data = pickle.load(f, encoding="bytes")
    f.close()
    return data  # (X_train, y_train), (X_test, y_test)


def load_fashion_mnist(
    train_img_path="train-images-idx3-ubyte.gz",
    train_label_path="train-labels-idx1-ubyte.gz",
    test_img_path="t10k-images-idx3-ubyte.gz",
    test_label_path="t10k-labels-idz1-ubyte.gz",
):
    train_img_path = get_file(train_img_path, origin=FA_TR_IMG_URL)
    train_label_path = get_file(train_label_path, origin=FA_TR_LBL_URL)
    test_img_path = get_file(test_img_path, origin=FA_TE_IMG_URL)
    test_label_path = get_file(test_label_path, origin=FA_TE_LBL_URL)
    # Based on https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py
    with gzip.open(train_label_path, "rb") as trlbpath:
        train_labels = np.frombuffer(trlbpath.read(), dtype=np.uint8, offset=8)
    with gzip.open(train_img_path, "rb") as trimgpath:
        train_images = np.frombuffer(
            trimgpath.read(), dtype=np.uint8, offset=16
        ).reshape(len(train_labels), 28, 28)
    with gzip.open(test_label_path, "rb") as telbpath:
        test_labels = np.frombuffer(telbpath.read(), dtype=np.uint8, offset=8)
    with gzip.open(test_img_path, "rb") as teimgpath:
        test_images = np.frombuffer(
            teimgpath.read(), dtype=np.uint8, offset=16
        ).reshape(len(test_labels), 28, 28)
    return (train_images, train_labels), (test_images, test_labels)


def load_kuzushiji_mnist(
    train_img_path="kmnist-train-imgs.npz",
    train_label_path="kmnist-train-labels.npz",
    test_img_path="kmnist-test-imgs.npz",
    test_label_path="kmnist-test-labels.npz",
):
    train_img_path = get_file(train_img_path, origin=KU_TR_IMG_URL)
    train_label_path = get_file(train_label_path, origin=KU_TR_LBL_URL)
    test_img_path = get_file(test_img_path, origin=KU_TE_IMG_URL)
    test_label_path = get_file(test_label_path, origin=KU_TE_LBL_URL)
    train_images = np.load(train_img_path)['arr_0']
    train_labels = np.load(train_label_path)['arr_0']
    test_images = np.load(test_img_path)['arr_0']
    test_labels = np.load(test_label_path)['arr_0']
    return (train_images, train_labels), (test_images, test_labels)
