from typing import Callable, Iterable
from functools import partial
from pathlib import Path
import random

import spacy
from spacy.training import Example
from .util import get_file
from ._registry import register_loader

IMDB_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


@spacy.registry.readers("ml_datasets.imdb_sentiment.v1")
def cmu_reader(
    path: Path, train: bool, limit: int = 0,
) -> Callable[["Language"], Iterable[Example]]:
    return partial(imdb, path, limit, train)


@register_loader("imdb")
def imdb(loc=None, limit=0, train=True, nlp=None):
    if loc is None:
        loc = get_file("aclImdb", IMDB_URL, untar=True, unzip=True)
    data_loc = Path(loc) / "test"
    if train:
        data_loc = Path(loc) / "train"
    return read_imdb(data_loc, nlp, limit=limit)


def read_imdb(data_dir, nlp, limit=0):
    locs = []
    for subdir in ("pos", "neg"):
        for filename in (data_dir / subdir).iterdir():
            locs.append((filename, subdir))

    # shuffle and filter the file locations
    random.shuffle(locs)
    if limit >= 1:
        locs = locs[:limit]

    examples = []
    for loc, gold_label in locs:
        with loc.open("r", encoding="utf8") as file_:
            text = file_.read()
        text = text.replace("<br />", "\n\n")
        if text.strip():
            doc = nlp.make_doc(text)
            cat_dict = {label: gold_label == label for label in ["pos", "neg"]}
            examples.append(Example.from_dict(doc, {"cats": cat_dict}))
    return examples
