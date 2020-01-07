import random
import gzip
from srsly import cloudpickle as pickle

from .util import unzip, to_categorical, get_file
from ._registry import register_loader


URL = "https://s3.amazonaws.com/img-datasets/mnist.pkl.gz"


@register_loader("mnist")
def mnist():
    # the data, shuffled and split between tran and test sets
    (X_train, y_train), (X_test, y_test) = load_mnist()
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train /= 255.0
    X_test /= 255.0
    train_data = list(zip(X_train, y_train))
    nr_train = X_train.shape[0]
    random.shuffle(train_data)
    heldout_data = train_data[: int(nr_train * 0.1)]
    mnist_train = train_data[len(heldout_data) :]
    mnist_dev = list(zip(X_test, y_test))

    train_X, train_Y = unzip(mnist_train)
    train_Y = to_categorical(train_Y, n_classes=10)
    dev_X, dev_Y = unzip(mnist_dev)
    dev_Y = to_categorical(dev_Y, n_classes=10)
    return (train_X, train_Y), (dev_X, dev_Y)


def load_mnist(path="mnist.pkl.gz"):
    path = get_file(path, origin=URL)
    if path.endswith(".gz"):
        f = gzip.open(path, "rb")
    else:
        f = open(path, "rb")
    data = pickle.load(f, encoding="bytes")
    f.close()
    return data  # (X_train, y_train), (X_test, y_test)
