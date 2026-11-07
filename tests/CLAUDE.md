# Tests

- Every client test runs on every registered backend: use `client` or a `client_*` fixture from `conftest.py`.
- A file holds one behaviour: table, reads, writes, locks, ids, oplog, stress; never one per backend or per coverage target.
- Behaviour only one backend has goes in `test_bigtable.py` or `test_hbase.py`.
- One behaviour, one test: parametrize variants, never copy them.
- Setup lives once, in `helpers.py` or `conftest.py`.
- Assert exact values: whole key sets and counts, never `>=`, never a loop that passes on an empty result.
- A test that cannot fail when its behaviour breaks is deleted.
- Names state the behaviour; comments state invariants, never what the code does.
- Every line a change adds under `kvdbclient/` is reached by a test, as Codecov's patch check counts it.
