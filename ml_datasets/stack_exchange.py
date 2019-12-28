from srsly import json_loads

from .util import partition


def stack_exchange(loc=None):
    if loc is None:
        raise ValueError("No default path for Stack Exchange yet")
    rows = []
    with loc.open("r", encoding="utf8") as file_:
        for line in file_:
            eg = json_loads(line)
            rows.append(((eg["text1"], eg["text2"]), int(eg["label"])))
    train, dev = partition(rows, 0.7)
    return train, dev
