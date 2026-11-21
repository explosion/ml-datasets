Fix spaCy readers and tarball extraction on newer Pythons

## Fixes

- Fix the spaCy readers (`ml_datasets.imdb_sentiment.v1`, `ml_datasets.dbpedia.v1`, `ml_datasets.cmu_movies.v1`) failing config validation when no `path` is given, with confection 1.3.4+
- Extract downloaded tarballs with the `"data"` extraction filter, avoiding the `DeprecationWarning` on Python 3.12+ and the default change in Python 3.14. On Pythons without extraction filters, archives with links, special files or paths outside the target directory are refused

## Other

- Run the test workflow on pushes to `master`
- Remove the `publish_pypi` workflow; publishing is handled by a separate release process
