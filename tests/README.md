# Tests for `AdaParse`

Test files should take the form `*_test.py` and tests inside files should be top-level functions named `test_*()`.

Tests should typically:
- test a single aspect of the code (e.g., to test feature `a` and `b`, use two separate test functions),
- only test the interface (e.g., tests should not check internal implementation details), and
- tests should not rely on the order in which they are executed.

## Data Availability
Check for weights of Nougat and Swin Transformer
```
ADAPARSE_RUN_ONLINE_TESTS=1 pytest -m online -q -vv -rA
```
Check if data is in the right place:
```
pytest -m fs -vv --log-cli-level=INFO
```

# Model Consistency
Check the standalone Swin encoder and multilingual BART decoder that make up Nougat
```
pytest test_standalone_swin_encoder.py -m fs -vv --log-cli-level=INFO -rs
```

BartDecoder via
```
pytest test_bart_decoder.py -m fs -vv --log-cli-level=INFO
```

Nougat via new model pipeline
```
pytest tests/test_nougat.py -m fs -vv --log-cli-level=INFO
```
