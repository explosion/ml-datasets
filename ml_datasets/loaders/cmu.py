import json
from pathlib import Path
import random
import csv

from ..util import get_file
from .._registry import register_loader

CMU_URL = "http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz"


@register_loader("cmu")
def cmu(train=True, loc=None, *, limit=0, shuffle=True):
    if loc is None:
        loc = get_file("MovieSummaries", CMU_URL, untar=True, unzip=True)
    meta_loc = Path(loc) / "movie.metadata.tsv"
    text_loc = Path(loc) / "plot_summaries.txt"
    return read_cmu(train, meta_loc, text_loc, limit=limit, shuffle=shuffle)


def read_cmu(train, meta_loc, text_loc, *, limit, shuffle):
    """Movies with an ID ending on 3, are considered to be test articles"""
    genre_by_id = {}
    title_by_id = {}
    examples = []
    with meta_loc.open("r", encoding="utf8") as file_:
        for row in csv.reader(file_, delimiter="\t"):
            movie_id = row[0]
            title = row[2]
            annot = row[8]
            d = json.loads(annot)
            genres = set(d.values())
            genre_by_id[movie_id] = genres
            title_by_id[movie_id] = title

    with text_loc.open("r", encoding="utf8") as file_:
        for row in csv.reader(file_, delimiter="\t"):
            movie_id = row[0]
            text = row[1]
            genres = genre_by_id.get(movie_id, None)
            title = title_by_id.get(movie_id, "")
            # only use examples with True cases in the final labels that made the frequency cut
            if genres:
                if train != str(movie_id).endswith("3"):
                    examples.append((title + "\n" + text, list(genres)))
    if shuffle:
        random.shuffle(examples)
    if limit >= 1:
        examples = examples[:limit]
    return examples
