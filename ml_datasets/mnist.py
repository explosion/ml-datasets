import random

from ._registry import register_loader


@register_loader("mnist")
def mnist():
    from ._vendorized.keras_datasets import load_mnist

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
    train_data = train_data[len(heldout_data) :]
    test_data = list(zip(X_test, y_test))
    return train_data, heldout_data, test_data
