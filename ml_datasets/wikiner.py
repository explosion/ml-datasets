from ._registry import register_loader


@register_loader("wikiner")
def wikiner(file_, tagmap=None):
    Xs = []
    ys = []
    for line in file_:
        if not line.strip():
            continue
        tokens = [t.rsplit("|", 2) for t in line.split()]
        words, _, tags = zip(*tokens)
        if tagmap is not None:
            tags = [tagmap.setdefault(tag, len(tagmap)) for tag in tags]
        Xs.append(words)
        ys.append(tags)
    return zip(Xs, ys)
