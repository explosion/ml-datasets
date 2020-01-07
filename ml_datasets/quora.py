from pathlib import Path
import csv

from .util import partition, get_file
from ._registry import register_loader


QUORA_QUESTIONS_URL = "http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv"


@register_loader("quora_questions")
def quora_questions(loc=None):
    if loc is None:
        loc = get_file("quora_similarity.tsv", QUORA_QUESTIONS_URL)
    if isinstance(loc, str):
        loc = Path(loc)
    is_header = True
    lines = []
    with loc.open("r", encoding="utf8") as file_:
        for row in csv.reader(file_, delimiter="\t"):
            if is_header:
                is_header = False
                continue
            id_, qid1, qid2, sent1, sent2, is_duplicate = row
            if not isinstance(sent1, str):
                sent1 = sent1.decode("utf8").strip()
            if not isinstance(sent2, str):
                sent2 = sent2.decode("utf8").strip()
            if sent1 and sent2:
                lines.append(((sent1, sent2), int(is_duplicate)))
    train, dev = partition(lines, 0.9)
    return train, dev
