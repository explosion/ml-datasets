from typing import Iterable, Callable
from pathlib import Path

from ..loaders import imdb


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
