import numpy
import tarfile
import zipfile
import os
import shutil
from urllib.error import URLError, HTTPError
from urllib.request import urlretrieve
import tqdm


def safe_extract_tar(tfile, path):
    """Extract a tar archive into `path`, refusing members that would be
    written outside of it.

    Uses the "data" extraction filter where Python supports it (3.12+ and
    the security backports in 3.10.12+ and 3.11.4+), which
    also avoids the DeprecationWarning that `extractall()` without a filter
    raises on Python 3.12+. On Pythons without extraction filters, links and
    special files are refused, and so are paths that leave `path`.
    """
    if hasattr(tarfile, "data_filter"):
        tfile.extractall(path=path, filter="data")
        return
    root = os.path.realpath(path)
    for member in tfile.getmembers():
        if not (member.isfile() or member.isdir()):
            raise tarfile.TarError(
                "Refusing to extract {!r}: links and special files are not "
                "supported on this Python version".format(member.name)
            )
        target = os.path.realpath(os.path.join(root, member.name))
        if os.path.commonpath([root, target]) != root:
            raise tarfile.TarError(
                "Refusing to extract {!r}: path is outside of the target "
                "directory".format(member.name)
            )
    tfile.extractall(path=path)


def get_file(fname, origin, untar=False, unzip=False, cache_subdir="datasets"):
    """Downloads a file from a URL if it not already in the cache."""
    # https://raw.githubusercontent.com/fchollet/keras/master/keras/utils/data_utils.py
    # Copyright Francois Chollet, Google, others (2015)
    # Under MIT license
    datadir_base = os.path.expanduser(os.path.join("~", ".keras"))
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join("/tmp", ".keras")
    datadir = os.path.join(datadir_base, cache_subdir)
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    untar_fpath = os.path.join(datadir, fname)
    if untar or unzip:
        if unzip:
            fpath = untar_fpath + ".zip"
        else:
            fpath = untar_fpath + ".tar.gz"
    else:
        fpath = os.path.join(datadir, fname)
    global progbar
    progbar = None

    def dl_progress(count, block_size, total_size):
        global progbar
        if progbar is None:
            progbar = tqdm.tqdm(total=total_size)
        else:
            progbar.update(block_size)

    if not os.path.exists(fpath):
        try:
            try:
                urlretrieve(origin, fpath, dl_progress)
            # Enrich download exceptions with full file name
            # HTTPError is a subclass of URLError, so it must be caught first
            except HTTPError as e:
                error_msg = "URL fetch failure on {} : {} -- {}"
                e.msg = error_msg.format(origin, e.code, e.msg)
                raise
            except URLError as e:
                error_msg = "URL fetch failure on {} -- {}"
                e.reason = error_msg.format(origin, e.reason)
                raise
        except (Exception, KeyboardInterrupt):
            if os.path.exists(fpath):
                os.remove(fpath)
            raise
        progbar = None

    if untar:
        if not os.path.exists(untar_fpath):
            print("Untaring file...")
            with tarfile.open(fpath, "r:gz") as tfile:
                try:
                    safe_extract_tar(tfile, datadir)
                except (Exception, KeyboardInterrupt):
                    if os.path.exists(untar_fpath):
                        if os.path.isfile(untar_fpath):
                            os.remove(untar_fpath)
                        else:
                            shutil.rmtree(untar_fpath)
                    raise
        return untar_fpath
    elif unzip:
        if not os.path.exists(untar_fpath):
            print("Unzipping file...")
            with zipfile.ZipFile(fpath) as file_:
                try:
                    file_.extractall(path=datadir)
                except (Exception, KeyboardInterrupt):
                    if os.path.exists(untar_fpath):
                        if os.path.isfile(untar_fpath):
                            os.remove(untar_fpath)
                        else:
                            shutil.rmtree(untar_fpath)
                    raise
        return untar_fpath
    return fpath


def partition(examples, split_size):
    examples = list(examples)
    numpy.random.shuffle(examples)
    n_docs = len(examples)
    split = int(n_docs * split_size)
    return examples[:split], examples[split:]


def unzip(data):
    x, y = zip(*data)
    return numpy.asarray(x), numpy.asarray(y)


def to_categorical(Y, n_classes=None):
    # From keras
    Y = numpy.array(Y, dtype="int").ravel()
    if not n_classes:
        n_classes = numpy.max(Y) + 1
    n = Y.shape[0]
    categorical = numpy.zeros((n, n_classes), dtype="float32")
    categorical[numpy.arange(n), Y] = 1
    return numpy.asarray(categorical)
