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
