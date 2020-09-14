from typing import Iterable, Callable
from pathlib import Path
from spacy.training.example import Example

from ..loaders import dbpedia


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
