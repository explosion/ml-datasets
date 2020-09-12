from typing import Callable, Iterable
from functools import partial
import json
from pathlib import Path
import random
import csv
import spacy
from spacy import Language
from spacy.training import Example

from .util import get_file
from ._registry import register_loader

CMU_URL = "http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz"


@spacy.registry.readers("ml_datasets.cmu_movies.v1")
def cmu_reader(
    path: Path, train: bool, freq_cutoff: int = 1000, limit: int = 0,
) -> Callable[["Language"], Iterable[Example]]:
    return partial(cmu, path, limit, train, freq_cutoff)


@register_loader("cmu")
def cmu(loc=None, limit=0, train=True, freq_cutoff=1000, nlp=None):
    if loc is None:
        loc = get_file("MovieSummaries", CMU_URL, untar=True, unzip=True)
    meta_loc = Path(loc) / "movie.metadata.tsv"
    text_loc = Path(loc) / "plot_summaries.txt"
    return read_cmu(meta_loc, text_loc, nlp, limit=limit, train=train, freq_cutoff=freq_cutoff)


def read_cmu(meta_loc, text_loc, nlp, limit, train, freq_cutoff, shuffle=True):
    examples = []
    genre_by_id = {}
    title_by_id = {}
    unique_genres = {}
    with meta_loc.open("r", encoding="utf8") as file_:
        for row in csv.reader(file_, delimiter="\t"):
            movie_id = row[0]
            title = row[2]
            annot = row[8]
            d = json.loads(annot)
            genres = set(d.values())
            for g in genres:
                unique_genres[g] = unique_genres.get(g, 0) + 1
            genre_by_id[movie_id] = genres
            title_by_id[movie_id] = title

    # filter labels by frequency
    final_labels = [l for l in sorted(unique_genres.keys()) if unique_genres[l] >= freq_cutoff]

    with text_loc.open("r", encoding="utf8") as file_:
        for row in csv.reader(file_, delimiter="\t"):
            movie_id = row[0]
            text = row[1]
            genres = genre_by_id.get(movie_id, None)
            title = title_by_id.get(movie_id, "")
            # only use examples with True cases in the final labels that made the frequency cut
            if genres and [l for l in genres if l in final_labels]:
                if train != str(movie_id).endswith("3"):
                    doc = nlp.make_doc(title + "\n" + text)
                    cat_dict = {label: label in genres for label in final_labels}
                    examples.append(Example.from_dict(doc, {"cats": cat_dict}))
    if shuffle:
        random.shuffle(examples)
    if limit >= 1:
        examples = examples[:limit]
    return examples
