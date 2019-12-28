from srsly import cloudpickle as pickle

from ._registry import register_loader
from ._vendorized.keras_data_utils import get_file


@register_loader("reuters")
def reuters():
    from ._vendorized.keras_datasets import load_reuters

    (X_train, y_train), (X_test, y_test) = load_reuters()
    return (X_train, y_train), (X_test, y_test)


def get_word_index(path="reuters_word_index.pkl"):
    path = get_file(
        path, origin="https://s3.amazonaws.com/text-datasets/reuters_word_index.pkl"
    )
    f = open(path, "rb")
    data = pickle.load(f, encoding="latin1")
    f.close()
    return data
