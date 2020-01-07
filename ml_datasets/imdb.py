from pathlib import Path
import random

from .util import get_file
from ._registry import register_loader

IMDB_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


@register_loader("imdb")
def imdb(loc=None, limit=0):
    if loc is None:
        loc = get_file("aclImdb", IMDB_URL, untar=True, unzip=True)
    train_loc = Path(loc) / "train"
    test_loc = Path(loc) / "test"
    return read_imdb(train_loc, limit=limit), read_imdb(test_loc, limit=limit)


def read_imdb(data_dir, limit=0):
    examples = []
    for subdir, label in (("pos", 1), ("neg", 0)):
        for filename in (data_dir / subdir).iterdir():
            with filename.open("r", encoding="utf8") as file_:
                text = file_.read()
            text = text.replace("<br />", "\n\n")
            if text.strip():
                examples.append((text, label))
    random.shuffle(examples)
    if limit >= 1:
        examples = examples[:limit]
    return examples
