import pytest
from urllib.error import HTTPError, URLError
from ml_datasets.util import get_file


def test_get_file_domain_resolution_fails():
    with pytest.raises(
        URLError, match=r"test_non_existent_file.*(not known|getaddrinfo failed)"
    ):
        get_file(
            "non_existent_file.txt",
            "http://test_notexist.wth/test_non_existent_file.txt"
        )


def test_get_file_404_file_not_found():
    with pytest.raises(HTTPError, match="test_non_existent_file.*404.*Not Found") as e:
        get_file(
            "non_existent_file.txt",
            "http://google.com/test_non_existent_file.txt"
        )
    assert e.value.code == 404
    # Suppress pytest.PytestUnraisableExceptionWarning:
    #          Exception ignored while calling deallocator
    # This questionable design quirk comes from urllib.request.urlretrieve,
    # so we shouldn't shim around it.
    e.value.close()


def _make_tar(path, members):
    import io
    import tarfile

    with tarfile.open(path, "w:gz") as tfile:
        for name, kind in members:
            info = tarfile.TarInfo(name)
            if kind == "file":
                data = b"hello"
                info.size = len(data)
                tfile.addfile(info, io.BytesIO(data))
            elif kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = "/etc/passwd"
                tfile.addfile(info)


def test_safe_extract_tar_extracts_without_warning(tmp_path, recwarn):
    import tarfile
    from ml_datasets.util import safe_extract_tar

    archive = tmp_path / "data.tar.gz"
    _make_tar(archive, [("data/a.txt", "file")])
    out = tmp_path / "out"
    with tarfile.open(archive, "r:gz") as tfile:
        safe_extract_tar(tfile, str(out))
    assert (out / "data" / "a.txt").read_bytes() == b"hello"
    assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]


@pytest.mark.parametrize(
    "members", [[("../evil.txt", "file")], [("data/link", "symlink")]]
)
def test_safe_extract_tar_refuses_escaping_members(tmp_path, members):
    import tarfile
    from ml_datasets.util import safe_extract_tar

    archive = tmp_path / "data.tar.gz"
    _make_tar(archive, members)
    out = tmp_path / "out"
    with tarfile.open(archive, "r:gz") as tfile:
        with pytest.raises(tarfile.TarError):
            safe_extract_tar(tfile, str(out))
    assert not (tmp_path / "evil.txt").exists()
    assert not (out / "data" / "link").exists()
