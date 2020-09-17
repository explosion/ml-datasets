from functools import partial
from typing import Iterable, Callable, Dict
from pathlib import Path

from .loaders import cmu, dbpedia, imdb


def cmu_reader(
    path: Path = None, *, freq_cutoff: int = 0, limit: int = 0, split=0.9
) -> Dict[str, Callable[["Language"], Iterable["Example"]]]:
    from spacy.training.example import Example

    # Deduce the categories above threshold by inspecting all data
    all_train_data, _ = list(cmu(path, limit=0, split=1))
    counted_cats = {}
    for text, cats in all_train_data:
        for cat in cats:
            counted_cats[cat] = counted_cats.get(cat, 0) + 1
    # filter labels by frequency
    unique_labels = [
        l for l in sorted(counted_cats.keys()) if counted_cats[l] >= freq_cutoff
    ]
    train_data, dev_data = cmu(path, limit=limit, shuffle=False, labels=unique_labels, split=split)

    def read_examples(data, nlp):
        for text, cats in data:
            doc = nlp.make_doc(text)
            assert isinstance(cats, list)
            cat_dict = {label: float(label in cats) for label in unique_labels}
            yield Example.from_dict(doc, {"cats": cat_dict})

    return {
        "train": partial(read_examples, train_data),
        "dev": partial(read_examples, dev_data),
    }


def dbpedia_reader(
    path: Path = None, *, train_limit: int = 0, dev_limit: int = 0
) -> Dict[str, Callable[["Language"], Iterable["Example"]]]:
    from spacy.training.example import Example

    all_train_data, _ = dbpedia(path, train_limit=0, dev_limit=1)
    unique_labels = set()
    for text, gold_label in all_train_data:
        assert isinstance(gold_label, str)
        unique_labels.add(gold_label)
    train_data, dev_data = dbpedia(path, train_limit=train_limit, dev_limit=dev_limit)

    def read_examples(data, nlp):
        for text, gold_label in data:
            doc = nlp.make_doc(text)
            cat_dict = {label: float(gold_label == label) for label in unique_labels}
            yield Example.from_dict(doc, {"cats": cat_dict})

    return {
        "train": partial(read_examples, train_data),
        "dev": partial(read_examples, dev_data),
    }


def imdb_reader(
    path: Path = None, *, train_limit: int = 0, dev_limit: int = 0
) -> Dict[str, Callable[["Language"], Iterable["Example"]]]:
    from spacy.training.example import Example

    train_data, dev_data = imdb(path, train_limit=train_limit, dev_limit=dev_limit)
    unique_labels = ["pos", "neg"]

    def read_examples(data, nlp):
        for text, gold_label in data:
            doc = nlp.make_doc(text)
            cat_dict = {label: float(gold_label == label) for label in unique_labels}
            yield Example.from_dict(doc, {"cats": cat_dict})

    return {
        "train": partial(read_examples, train_data),
        "dev": partial(read_examples, dev_data),
    }
