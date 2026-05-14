import logging

from libs.common.schemas import ActionType
from services.policy_layer.preflop_charts import expand_range, get_preflop_action


def test_expand_range():
    res = expand_range("AA, AKs, 22")
    assert "AA" in res
    assert "AKs" in res
    assert "22" in res

    res2 = expand_range("22+")
    assert "AA" in res2
    assert "22" in res2
    assert "77" in res2


def test_get_preflop_action():
    # 6-max UTG AA should RAISE
    assert get_preflop_action(6, "UTG", "AA") == ActionType.RAISE

    # 6-max UTG 72o should not be in chart (None)
    assert get_preflop_action(6, "UTG", "72o") is None

    # BB calling hands
    assert get_preflop_action(6, "BB", "87o") == ActionType.CALL


def test_expand_range_ax_suited_open_regression():
    """Regression: A2s+ used to be silently dropped by `expand_range` because
    of an embedded `# noqa: E501` token. Ensure the full Ax-suited family is
    expanded and the corresponding chart lookups succeed.
    """
    res = expand_range("A2s+")
    for r in "23456789TJQK":
        assert f"A{r}s" in res, f"Missing A{r}s in expanded range"

    # BTN and CO opening A2s should now return a non-None chart action.
    assert get_preflop_action(6, "BTN", "A2s") is not None
    assert get_preflop_action(6, "CO", "A2s") is not None
    assert get_preflop_action(6, "SB", "A2s") is not None


def test_expand_range_dash_token():
    """Regression: dash-form ranges like '77-99' should expand inclusively."""
    res = expand_range("77-99")
    assert {"77", "88", "99"} == res


def test_expand_range_logs_unknown_token(caplog):
    """Malformed tokens previously vanished silently; they must now warn."""
    with caplog.at_level(logging.WARNING, logger="services.policy_layer.preflop_charts"):
        res = expand_range("AA, #noqa:E501, FOO")
    assert res == {"AA"}
    assert any("ignoring" in record.message for record in caplog.records)
