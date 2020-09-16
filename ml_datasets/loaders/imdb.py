from pathlib import Path
import random

from ..util import get_file
from .._registry import register_loader

IMDB_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


@register_loader("imdb")
def imdb(loc=None, *, train_limit=0, dev_limit=0):
    if loc is None:
        loc = get_file("aclImdb", IMDB_URL, untar=True, unzip=True)
    train_loc = Path(loc) / "train"
    test_loc = Path(loc) / "test"
    return read_imdb(train_loc, limit=train_limit), read_imdb(test_loc, limit=dev_limit)


def read_imdb(data_dir, *, limit=0):
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
            examples.append((text, gold_label))
    return examples
