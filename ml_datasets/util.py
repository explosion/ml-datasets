import numpy
import tarfile
import zipfile
import os
import shutil
from urllib.error import URLError, HTTPError
from urllib.request import urlretrieve
import tqdm


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
    if untar or unzip:
        untar_fpath = os.path.join(datadir, fname)
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

    error_msg = "URL fetch failure on {}: {} -- {}"
    if not os.path.exists(fpath):
        try:
            try:
                urlretrieve(origin, fpath, dl_progress)
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
        except (Exception, KeyboardInterrupt):
            if os.path.exists(fpath):
                os.remove(fpath)
            raise
        progbar = None

    if untar:
        if not os.path.exists(untar_fpath):
            print("Untaring file...")
            tfile = tarfile.open(fpath, "r:gz")
            try:
                tfile.extractall(path=datadir)
            except (Exception, KeyboardInterrupt):
                if os.path.exists(untar_fpath):
                    if os.path.isfile(untar_fpath):
                        os.remove(untar_fpath)
                    else:
                        shutil.rmtree(untar_fpath)
                raise
            tfile.close()
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
