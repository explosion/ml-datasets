import numpy
from collections import Counter
from pathlib import Path

from .util import get_file
from ._registry import register_loader


GITHUB = "https://github.com/UniversalDependencies/"
ANCORA_1_4_ZIP = "{github}/{ancora}/archive/r1.4.zip".format(
    github=GITHUB, ancora="UD_Spanish-AnCora"
)
EWTB_1_4_ZIP = "{github}/{ewtb}/archive/r1.4.zip".format(
    github=GITHUB, ewtb="UD_English"
)


@register_loader("ud_ancora_pos_tags")
def ud_ancora_pos_tags(encode_words=False):
    data_dir = Path(get_file("UD_Spanish-AnCora-r1.4", ANCORA_1_4_ZIP, unzip=True))
    train_loc = data_dir / "es_ancora-ud-train.conllu"
    dev_loc = data_dir / "es_ancora-ud-dev.conllu"
    return ud_pos_tags(train_loc, dev_loc, encode_words=encode_words)


@register_loader("ud_ewtb_pos_tags")
def ud_ewtb_pos_tags(encode_tags=False, encode_words=False):
    data_dir = Path(get_file("UD_English-EWT-r1.4", EWTB_1_4_ZIP, unzip=True))
    train_loc = data_dir / "en-ud-train.conllu"
    dev_loc = data_dir / "en-ud-dev.conllu"
    return ud_pos_tags(
        train_loc, dev_loc, encode_tags=encode_tags, encode_words=encode_words
    )


def ud_pos_tags(train_loc, dev_loc, encode_tags=True, encode_words=True):
    train_sents = list(read_conll(train_loc))
    dev_sents = list(read_conll(dev_loc))
    tagmap = {}
    freqs = Counter()
    for words, tags in train_sents:
        for tag in tags:
            tagmap.setdefault(tag, len(tagmap))
        for word in words:
            freqs[word] += 1
    vocab = {
        word: i for i, (word, freq) in enumerate(freqs.most_common()) if (freq >= 5)
    }

    def _encode(sents):
        X = []
        y = []
        for words, tags in sents:
            if encode_words:
                X.append(
                    numpy.asarray(
                        [vocab.get(word, len(vocab)) for word in words], dtype="uint64"
                    )
                )
            else:
                X.append(words)
            if encode_tags:
                y.append(numpy.asarray([tagmap[tag] for tag in tags], dtype="int32"))
            else:
                y.append(tags)
        return zip(X, y)

    return _encode(train_sents), _encode(dev_sents), len(tagmap)


def read_conll(loc):
    with Path(loc).open(encoding="utf8") as file_:
        sent_strs = file_.read().strip().split("\n\n")
    for sent_str in sent_strs:
        lines = [
            line.split() for line in sent_str.split("\n") if not line.startswith("#")
        ]
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
