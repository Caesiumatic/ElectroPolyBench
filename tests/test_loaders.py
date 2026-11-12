"""Dataset selection must survive download failures without changing the experiment."""

import pytest

from data import loaders


def test_panda_failure_propagates(monkeypatch):
    def unavailable(**kwargs):
        raise OSError("PANDA unavailable")

    monkeypatch.setattr(loaders, "load_panda", unavailable)
    with pytest.raises(OSError, match="PANDA unavailable"):
        loaders.load_dataset(verbose=False)


def test_explicit_synthetic_runs_offline(monkeypatch):
    def unexpected_download(**kwargs):
        pytest.fail("synthetic selection requested a download")

    monkeypatch.setattr(loaders, "load_panda", unexpected_download)
    df, info = loaders.load_dataset(prefer_real=False, verbose=False)
    assert len(df) == info["n"] == 216
    assert info["is_real"] is False
    assert set(df["source"]) == {"synthetic"}
