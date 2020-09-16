from pathlib import Path
import csv
import random

from ..util import get_file
from .._registry import register_loader


# DBPedia Ontology from https://course.fast.ai/datasets
DBPEDIA_ONTOLOGY_URL = "https://s3.amazonaws.com/fast-ai-nlp/dbpedia_csv.tgz"


@register_loader("dbpedia")
def dbpedia(loc=None, *, train_limit=0, dev_limit=0):
    if loc is None:
        loc = get_file("dbpedia_csv", DBPEDIA_ONTOLOGY_URL, untar=True, unzip=True)
    train_loc = Path(loc) / "train.csv"
    test_loc = Path(loc) / "test.csv"
    return (
        read_dbpedia_ontology(train_loc, limit=train_limit),
        read_dbpedia_ontology(test_loc, limit=dev_limit),
    )


def read_dbpedia_ontology(data_file, *, limit=0):
    examples = []
    with open(data_file, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            label = row[0]
            title = row[1]
            text = row[2]
            examples.append((title + "\n" + text, label))
    random.shuffle(examples)
    if limit >= 1:
        examples = examples[:limit]
    return examples
