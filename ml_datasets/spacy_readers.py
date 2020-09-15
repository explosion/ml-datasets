from typing import Iterable, Callable
from pathlib import Path

from .loaders import cmu, dbpedia, imdb


def cmu_reader(
    train: bool, path: Path = None, *, freq_cutoff: int = 0, limit: int = 0
) -> Callable[["Language"], Iterable["Example"]]:
    from spacy.training.example import Example

    # Deduce the categories above threshold by inspecting all training data
    all_train_data = list(cmu(train=True, loc=path, limit=0))
    counted_cats = {}
    for text, cats in all_train_data:
        for cat in cats:
            counted_cats[cat] = counted_cats.get(cat, 0) + 1
    # filter labels by frequency
    unique_labels = [
        l for l in sorted(counted_cats.keys()) if counted_cats[l] >= freq_cutoff
    ]
    # do this here to avoid reading the data multiple times
    data = list(cmu(train, path, limit=limit, shuffle=False, labels=unique_labels))

    def read_examples(nlp):
        for text, cats in data:
            doc = nlp.make_doc(text)
            assert isinstance(cats, list)
            cat_dict = {label: float(label in cats) for label in unique_labels}
            yield Example.from_dict(doc, {"cats": cat_dict})

    return read_examples


def dbpedia_reader(
    train: bool, path: Path = None, *, limit: int = 0
) -> Callable[["Language"], Iterable["Example"]]:
    from spacy.training.example import Example

    all_train_data = dbpedia(train=True, loc=path, limit=0)
    unique_labels = set()
    for text, gold_label in all_train_data:
        assert isinstance(gold_label, str)
        unique_labels.add(gold_label)
    # do this here to avoid reading the data multiple times
    if train:
        data = all_train_data
        if limit >= 1:
            data = data[:limit]
    else:
        data = list(dbpedia(train, path, limit=limit))

    def read_examples(nlp):
        for text, gold_label in data:
            doc = nlp.make_doc(text)
            cat_dict = {label: float(gold_label == label) for label in unique_labels}
            yield Example.from_dict(doc, {"cats": cat_dict})

    return read_examples


def imdb_reader(
    train: bool, path: Path = None, *, limit: int = 0
) -> Callable[["Language"], Iterable["Example"]]:
    from spacy.training.example import Example

    # do this here to avoid reading the data multiple times
    data = list(imdb(train, path, limit=limit))
    unique_labels = ["pos", "neg"]

    def read_examples(nlp):
        for text, gold_label in data:
            doc = nlp.make_doc(text)
            cat_dict = {label: float(gold_label == label) for label in unique_labels}
            yield Example.from_dict(doc, {"cats": cat_dict})

    return read_examples
