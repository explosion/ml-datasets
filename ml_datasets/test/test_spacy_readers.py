import inspect

import pytest

from ml_datasets.spacy_readers import cmu_reader, dbpedia_reader, imdb_reader


@pytest.mark.parametrize("reader", [cmu_reader, dbpedia_reader, imdb_reader])
def test_reader_signature_accepts_default_path(reader):
    # spaCy resolves registered readers through confection, which builds a
    # pydantic model from the function signature and validates the arguments,
    # including defaults. With `path: Path = None`, pydantic v2 rejects the
    # default, so the readers fail unless an explicit path is given.
    pydantic = pytest.importorskip("pydantic")
    param = inspect.signature(reader).parameters["path"]
    model = pydantic.create_model("ReaderArgs", path=(param.annotation, param.default))
    assert model.model_validate({"path": None}).path is None
    assert model.model_validate({}).path is None
