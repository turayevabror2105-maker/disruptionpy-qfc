# Tests

```bash
pip install -e ".[dev]"
pytest -m "not slow and not integration"        # no data, a few minutes
DL_DATAFRAME=DL_DataFrame.csv pytest -m slow    # recomputes the worked example
DISRUPTIONPY_FRAME=frame.csv DPQ_TUD_COL=time_until_disrupt pytest -m integration
```

| group | needs | checks |
|---|---|---|
| default | nothing | the procedure against the paper's reference code, the attribution methods, the cap rule, the bootstrap, the validation and error messages, the numbers the README and these documents state against `docs/worked_example.json`, that every relative link between them resolves, that the record was made by the code now in `src/`, and that the version and release date agree across `pyproject.toml`, `__version__`, `CITATION.cff`, `CHANGELOG.md` and the install line |
| `slow` | the C-Mod release | the worked example reproduces its record |
| `integration` | a frame produced by DisruptionPy | the package accepts and screens a real DisruptionPy frame |

CI runs the default group on Python 3.10, 3.11 and 3.12, at the oldest dependency versions allowed, and from the
built wheel; it runs the integration group on a frame that DisruptionPy itself builds from MAST's open data
(`tests/fixtures/build_mast_frame.py`), where a missing frame is a failure, not a skip. That job runs on pushes to
`main`, weekly and on demand, not on pull requests, to keep load off the public data server. The slow group needs
the C-Mod release and is run before each release; its record is [`docs/worked_example.json`](worked_example.json).
[ATTACKS.md](ATTACKS.md) lists the strongest objections to the measurement and to this package, each with the test
or section that answers it.
