Fix spaCy readers and tarball extraction on newer Pythons

## Fixes

- Fix the spaCy readers (`ml_datasets.imdb_sentiment.v1`, `ml_datasets.dbpedia.v1`, `ml_datasets.cmu_movies.v1`) failing config validation when no `path` is given, with confection 1.3.4+
- Extract downloaded tarballs with the `"data"` extraction filter, avoiding the `DeprecationWarning` on Python 3.12+ and the default change in Python 3.14. On Pythons without extraction filters, archives with links, special files or paths outside the target directory are refused

## Other

- Require Python 3.10+ (Python 3.8 and 3.9 are end-of-life) and NumPy 1.21.2+, the first release with Python 3.10 wheels
- Test on Python 3.10 through 3.15
- Run the test workflow on pushes to `master`
- Skip dataset tests when the download host is unreachable, and run the spaCy reader tests in CI
- Remove the `publish_pypi` workflow; publishing is handled by a separate release process
