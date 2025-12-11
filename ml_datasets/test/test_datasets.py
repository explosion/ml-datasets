import pytest
import numpy as np

import ml_datasets

NP_VERSION = tuple(int(x) for x in np.__version__.split(".")[:2])

# FIXME warning on NumPy 2.4 when downloading pre-computed pickles:
# Python or NumPy boolean but got `align=0`.
# Did you mean to pass a tuple to create a subarray type? (Deprecated NumPy 2.4)
if NP_VERSION >= (2, 4):
    np_24_deprecation = pytest.mark.filterwarnings(
        "ignore::numpy.exceptions.VisibleDeprecationWarning", 
    
    )
else:
    # Note: can't use `condition=NP_VERSION >= (2, 4)` on the decorator directly
    # as numpy.exceptions did not exist in old NumPy versions.
    np_24_deprecation = lambda x: x


@np_24_deprecation
def test_cifar():
    (X_train, y_train), (X_test, y_test) = ml_datasets.cifar()
    # TODO test output contents


@pytest.mark.skip(reason="very slow download")
def test_cmu():
    train, dev = ml_datasets.cmu()
    # TODO test output contents


def test_dbpedia():
    train, dev = ml_datasets.dbpedia()
    # TODO test output contents


def test_imdb():
    train, dev = ml_datasets.imdb()
    # TODO test output contents


@np_24_deprecation
def test_mnist():
    (X_train, y_train), (X_test, y_test) = ml_datasets.mnist()
    # TODO test output contents


@pytest.mark.xfail(reason="403 Forbidden")
def test_quora_questions():
    train, dev = ml_datasets.quora_questions()
    # TODO test output contents


@np_24_deprecation
def test_reuters():
    (X_train, y_train), (X_test, y_test) = ml_datasets.reuters()
    # TODO test output contents


def test_snli():
    train, dev = ml_datasets.snli()
    # TODO test output contents


@pytest.mark.xfail(reason="no default path")
def test_stack_exchange():
    train, dev = ml_datasets.stack_exchange()
    # TODO test output contents


def test_ud_ancora_pos_tags():
    (train_X, train_y), (dev_X, dev_y) = ml_datasets.ud_ancora_pos_tags()
    # TODO test output contents


@pytest.mark.xfail(reason="str column where int expected")
def test_ud_ewtb_pos_tags():
    (train_X, train_y), (dev_X, dev_y) = ml_datasets.ud_ewtb_pos_tags()
    # TODO test output contents


@pytest.mark.xfail(reason="no default path")
def test_wikiner():
    train, dev = ml_datasets.wikiner()
    # TODO test output contents
