import numpy
from collections import Counter
from pathlib import Path

from .util import get_file, to_categorical
from ._registry import register_loader


GITHUB = "https://github.com/UniversalDependencies/"
TEMPLATE = "{github}/{repo}/archive/r1.4.zip"
ANCORA_1_4_ZIP = TEMPLATE.format(github=GITHUB, repo="UD_Spanish-AnCora")
EWTB_1_4_ZIP = TEMPLATE.format(github=GITHUB, repo="UD_English")


@register_loader("ud_ancora_pos_tags")
def ud_ancora_pos_tags(encode_words=False, limit=None):
    data_dir = Path(get_file("UD_Spanish-AnCora-r1.4", ANCORA_1_4_ZIP, unzip=True))
    train_loc = data_dir / "es_ancora-ud-train.conllu"
    dev_loc = data_dir / "es_ancora-ud-dev.conllu"
    return ud_pos_tags(train_loc, dev_loc, encode_words=encode_words, limit=limit)


@register_loader("ud_ewtb_pos_tags")
def ud_ewtb_pos_tags(encode_tags=False, encode_words=False, limit=None):
    data_dir = Path(get_file("UD_English-EWT-r1.4", EWTB_1_4_ZIP, unzip=True))
    train_loc = data_dir / "en-ud-train.conllu"
    dev_loc = data_dir / "en-ud-dev.conllu"
    return ud_pos_tags(
        train_loc,
        dev_loc,
        encode_tags=encode_tags,
        encode_words=encode_words,
        limit=limit,
    )


def ud_pos_tags(train_loc, dev_loc, encode_tags=True, encode_words=True, limit=None):
    train_sents = list(read_conll(train_loc))
    dev_sents = list(read_conll(dev_loc))
    tagmap = {}
    freqs = Counter()
    for words, tags in train_sents:
        for tag in tags:
            tagmap.setdefault(tag, len(tagmap))
        for word in words:
            freqs[word] += 1
    vocab = {w: i for i, (w, freq) in enumerate(freqs.most_common()) if (freq >= 5)}

    def _encode(sents):
        X = []
        y = []
        for words, tags in sents:
            if encode_words:
                arr = [vocab.get(word, len(vocab)) for word in words]
                X.append(numpy.asarray(arr, dtype="uint64"))
            else:
                X.append(words)
            if encode_tags:
                y.append(numpy.asarray([tagmap[tag] for tag in tags], dtype="int32"))
            else:
                y.append(tags)
        return zip(X, y)

    train_data = _encode(train_sents)
    check_data = _encode(dev_sents)
    train_X, train_y = zip(*train_data)
    dev_X, dev_y = zip(*check_data)
    nb_tag = max(max(y) for y in train_y) + 1
    train_X = list(train_X)
    dev_X = list(dev_X)
    train_y = [to_categorical(y, nb_tag) for y in train_y]
    dev_y = [to_categorical(y, nb_tag) for y in dev_y]
    if limit is not None:
        train_X = train_X[:limit]
        train_y = train_y[:limit]
    return (train_X, train_y), (dev_X, dev_y)


def read_conll(loc):
    with Path(loc).open(encoding="utf8") as file_:
        sent_strs = file_.read().strip().split("\n\n")
    for sent_str in sent_strs:
        lines = [li.split() for li in sent_str.split("\n") if not li.startswith("#")]
        words = []
        tags = []
        for i, pieces in enumerate(lines):
            if len(pieces) == 4:
                word, pos, head, label = pieces
            else:
                idx, word, lemma, pos1, pos, morph, head, label, _, _2 = pieces
            if "-" in idx:
                continue
            words.append(word)
            tags.append(pos)
        yield words, tags
