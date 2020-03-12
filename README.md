<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

# Machine learning dataset loaders

Loaders for various machine learning datasets for testing and example scripts.
Previously in `thinc.extra.datasets`.

[![PyPi Version](https://img.shields.io/pypi/v/ml-datasets.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.python.org/pypi/ml-datasets)

## Setup and installation

The package can be installed via pip:

```bash
pip install ml-datasets
```

## Loaders

Loaders can be imported directly or used via their string name (which is useful if they're set via command line arguments). Some loaders may take arguments – see the source of details.

```python
# Import directly
from ml_datasets import imdb
train_data, dev_data = imdb()
```

```python
# Load via registry
from ml_datasets import loaders
imdb_loader = loaders.get("imdb")
train_data, dev_data = imdb_loader()
```

### Available loaders

#### NLP datasets

| ID / Function        | Description                                  | NLP task                                  | From URL |
| -------------------- | -------------------------------------------- | ----------------------------------------- | :------: |
| `imdb`               | IMDB sentiment dataset                       | Binary classification: sentiment analysis |    ✓     |
| `dbpedia`            | DBPedia ontology dataset                     | Multi-label (exclusive) classification    |    ✓     |
| `quora_questions`    | Quora question answer dataset                |                                           |    ✓     |
| `reuters`            | Reuters dataset                              |                                           |    ✓     |
| `snli`               | Stanford Natural Language Inference corpus   |                                           |    ✓     |
| `stack_exchange`     | Stack Exchange dataset                       |                                           |          |
| `ud_ancora_pos_tags` | Universal Dependencies Spanish AnCora corpus | POS tagging                               |    ✓     |
| `ud_ewtb_pos_tags`   | Universal Dependencies English EWT corpus    | POS tagging                               |    ✓     |
| `wikiner`            | WikiNER data                                 |                                           |          |

#### Other ML datasets

| ID / Function        | Description                                  | ML task                                  | From URL |
| -------------------- | -------------------------------------------- | ----------------------------------------- | :------: |
| `mnist`              | MNIST data                                   |  Image recognition                                         |    ✓     |


### Dataset details

#### IMDB

Each instance contains the text of a movie review, and a sentiment expressed as `0` or `1`.

```python
train_data, dev_data = ml_datasets.imdb()
```

- Download URL: [http://ai.stanford.edu/~amaas/data/sentiment/](http://ai.stanford.edu/~amaas/data/sentiment/)
- Citation: [Andrew L. Maas et al., 2011](https://www.aclweb.org/anthology/P11-1015/)

| Property                      | Training         | Dev              |
| ----------------------------- | ---------------- | ---------------- |
| # Instances                   | 25000            | 25000            |
| Label values                  | {`0`, `1`}       | {`0`, `1`}       |
| Number of labels per instance | 1 (single)       | 1 (single)       |
| Label distribution            | balanced (50/50) | balanced (50/50) |

#### DBPedia

```python
train_data, dev_data = ml_datasets.dbpedia()
```

Each instance contains an ontological description, and a classification into one of the 14 distinct labels.

- Download URL: [Via fast.ai](https://s3.amazonaws.com/fast-ai-nlp/dbpedia_csv.tgz)
- Original citation: [Xiang Zhang et al., 2015](https://arxiv.org/abs/1509.01626)

| Property                      | Training   | Dev        |
| ----------------------------- | ---------- | ---------- |
| # Instances                   | 560000     | 70000      |
| Label values                  | `1`-`14`   | `1`-`14`   |
| Number of labels per instance | 1 (single) | 1 (single) |
| Label distribution            | balanced   | balanced   |

### Registering loaders

Loaders can be registered externally using the `loaders` registry as a decorator. For example:

```python
@ml_datasets.loaders("my_custom_loader")
def my_custom_loader():
    return load_some_data()

assert "my_custom_loader" in ml_datasets.loaders
```
