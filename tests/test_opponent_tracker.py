import pytest

from libs.common.schemas import ActionType, Street
from services.opponent_tracker.tracker import OpponentTracker

@pytest.fixture
def tracker():
    return OpponentTracker()

def test_initial_profile(tracker):
    profile = tracker.get_profile(seat_id=1)
    assert profile.seat_id == 1
    assert profile.hands_seen == 0
    assert profile.vpip == 0.0
    assert profile.pfr == 0.0
    assert profile.af == 0.0
    assert profile.fold_to_cbet == 0.0
    assert profile.sizing_tells == {}
    assert profile.sample_size == 0

def test_vpip_pfr_calculation(tracker):
    # Action 1: Call preflop (VPIP increases, PFR stays 0)
    tracker.record_action(seat_id=1, hand_id="h1", action=ActionType.CALL, sizing_ratio=0.0, time_to_act_ms=1000, street=Street.PREFLOP, position="BTN", won_showdown=False)
    profile = tracker.get_profile(seat_id=1)
    assert profile.vpip == 1.0
    assert profile.pfr == 0.0
    assert profile.hands_seen == 1

    # Action 2: Fold preflop (VPIP opportunity, but no action)
    tracker.record_action(seat_id=1, hand_id="h2", action=ActionType.FOLD, sizing_ratio=0.0, time_to_act_ms=1000, street=Street.PREFLOP, position="SB", won_showdown=False)
    profile = tracker.get_profile(seat_id=1)
    assert profile.vpip == 0.5  # 1/2
    assert profile.pfr == 0.0   # 0/2

    # Action 3: Raise preflop (Both VPIP and PFR increase)
    tracker.record_action(seat_id=1, hand_id="h3", action=ActionType.RAISE, sizing_ratio=3.0, time_to_act_ms=1000, street=Street.PREFLOP, position="BB", won_showdown=False)
    profile = tracker.get_profile(seat_id=1)
    assert profile.vpip == pytest.approx(0.666, abs=0.01) # 2/3
    assert profile.pfr == pytest.approx(0.333, abs=0.01)  # 1/3

def test_af_calculation(tracker):
    # Aggression Factor = Aggressive Actions / Passive Actions
    tracker.record_action(seat_id=2, hand_id="h1", action=ActionType.CALL, sizing_ratio=0.0, time_to_act_ms=1000, street=Street.FLOP, position="BTN", won_showdown=False)
    tracker.record_action(seat_id=2, hand_id="h1", action=ActionType.BET, sizing_ratio=0.5, time_to_act_ms=1000, street=Street.TURN, position="BTN", won_showdown=False)
    tracker.record_action(seat_id=2, hand_id="h1", action=ActionType.RAISE, sizing_ratio=2.0, time_to_act_ms=1000, street=Street.RIVER, position="BTN", won_showdown=False)

    profile = tracker.get_profile(seat_id=2)
    # 2 aggressive (BET, RAISE), 1 passive (CALL)
    assert profile.af == 2.0

    # Test AF with 0 passive actions
    tracker.record_action(seat_id=3, hand_id="h2", action=ActionType.BET, sizing_ratio=0.5, time_to_act_ms=1000, street=Street.FLOP, position="BTN", won_showdown=False)
    profile3 = tracker.get_profile(seat_id=3)
    assert profile3.af == float('inf')

def test_fold_to_cbet_calculation(tracker):
    # Action 1: Call on Flop (Faces cbet, doesn't fold)
    tracker.record_action(seat_id=4, hand_id="h1", action=ActionType.CALL, sizing_ratio=0.0, time_to_act_ms=1000, street=Street.FLOP, position="BB", won_showdown=False)
    profile = tracker.get_profile(seat_id=4)
    assert profile.fold_to_cbet == 0.0

    # Action 2: Fold on Flop (Faces cbet, folds)
    tracker.record_action(seat_id=4, hand_id="h2", action=ActionType.FOLD, sizing_ratio=0.0, time_to_act_ms=1000, street=Street.FLOP, position="BB", won_showdown=False)
    profile = tracker.get_profile(seat_id=4)
    assert profile.fold_to_cbet == 0.5  # 1 fold out of 2 cbets faced

def test_sizing_tells_and_reset(tracker):
    # Record some bets to generate sizing tells
    tracker.record_action(seat_id=5, hand_id="h1", action=ActionType.BET, sizing_ratio=0.5, time_to_act_ms=1000, street=Street.FLOP, position="BTN", won_showdown=False)
    tracker.record_action(seat_id=5, hand_id="h2", action=ActionType.BET, sizing_ratio=0.75, time_to_act_ms=1000, street=Street.FLOP, position="BTN", won_showdown=False)
    tracker.record_action(seat_id=5, hand_id="h3", action=ActionType.RAISE, sizing_ratio=2.0, time_to_act_ms=1000, street=Street.TURN, position="BTN", won_showdown=False)

    profile = tracker.get_profile(seat_id=5)

    # Check sizing tells averages
    assert profile.sizing_tells[Street.FLOP] == 0.625  # (0.5 + 0.75) / 2
    assert profile.sizing_tells[Street.TURN] == 2.0
    assert Street.RIVER not in profile.sizing_tells

    # Test reset
    tracker.reset_session()
    profile_after_reset = tracker.get_profile(seat_id=5)
    assert profile_after_reset.hands_seen == 0
    assert profile_after_reset.vpip == 0.0
    assert profile_after_reset.sizing_tells == {}
