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
