from srsly import json_loads
from pathlib import Path

from .util import get_file
from ._registry import register_loader


SNLI_URL = "http://nlp.stanford.edu/projects/snli/snli_1.0.zip"
THREE_LABELS = {"entailment": 2, "contradiction": 1, "neutral": 0}
TWO_LABELS = {"entailment": 1, "contradiction": 0, "neutral": 0}


@register_loader("snli")
def snli(loc=None, ternary=False):
    label_scheme = THREE_LABELS if ternary else TWO_LABELS
    if loc is None:
        loc = get_file("snli_1.0", SNLI_URL, unzip=True)
    if isinstance(loc, str):
        loc = Path(loc)
    train = read_snli(Path(loc) / "snli_1.0_train.jsonl", label_scheme)
    dev = read_snli(Path(loc) / "snli_1.0_dev.jsonl", label_scheme)
    return train, dev


def read_snli(loc, label_scheme):
    rows = []
    with loc.open("r", encoding="utf8") as file_:
        for line in file_:
            eg = json_loads(line)
            label = eg["gold_label"]
            if label == "-":
                continue
            rows.append(((eg["sentence1"], eg["sentence2"]), label_scheme[label]))
    return rows
