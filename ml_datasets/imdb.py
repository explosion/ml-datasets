from pathlib import Path
import random

from .util import get_file
from ._registry import register_loader

IMDB_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


@register_loader("imdb")
def imdb(train=True, loc=None, *, limit=0):
    if loc is None:
        loc = get_file("aclImdb", IMDB_URL, untar=True, unzip=True)
    data_loc = Path(loc) / "test"
    if train:
        data_loc = Path(loc) / "train"
    return read_imdb(data_loc, limit=limit)


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
