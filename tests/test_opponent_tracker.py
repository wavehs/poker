import pytest
from libs.common.schemas import ActionType, Street, TableState, PlayerState
from services.opponent_tracker.tracker import OpponentTracker

@pytest.fixture
def tracker():
    return OpponentTracker()

def test_record_action(tracker):
    player = PlayerState(seat=1, is_hero=False, is_active=True, last_action=ActionType.CALL, bet=10.0)
    state = TableState(
        is_hand_in_progress=True,
        street=Street.PREFLOP,
        pot=15.0,
        players=[player]
    )
    tracker.update(state)

    assert (1, Street.PREFLOP) in tracker.last_observed_action
    assert tracker.last_observed_action[(1, Street.PREFLOP)] == ActionType.CALL

    stats = tracker.profiles_raw[1]
    assert stats["pass_actions"] == 1
    assert stats["aggr_actions"] == 0

def test_get_profile(tracker):
    player = PlayerState(seat=2, is_hero=False, is_active=True, last_action=ActionType.RAISE, bet=20.0)
    state = TableState(
        is_hand_in_progress=True,
        street=Street.PREFLOP,
        pot=30.0,
        players=[player]
    )
    tracker.update(state)

    assert player.profile is not None
    assert player.profile.hands_played == 0
    assert player.profile.vpip == 1.0
    assert player.profile.pfr == 1.0
    assert player.profile.af == 1.0

def test_vpip_calculation(tracker):
    player = PlayerState(seat=3, is_hero=False, is_active=True, last_action=ActionType.CALL, bet=10.0)
    state = TableState(
        is_hand_in_progress=True,
        street=Street.PREFLOP,
        pot=15.0,
        players=[player]
    )
    tracker.update(state)

    stats = tracker.profiles_raw[3]
    assert stats["vpip_this_hand"] is True
    assert stats["vpip_hands"] == 1
    assert stats["pfr_this_hand"] is False
    assert stats["pfr_hands"] == 0
    assert player.profile.vpip == 1.0

def test_reset_session(tracker):
    player = PlayerState(seat=4, is_hero=False, is_active=True, last_action=ActionType.CALL, bet=10.0)
    state1 = TableState(
        is_hand_in_progress=True,
        street=Street.FLOP,
        pot=50.0,
        players=[player]
    )
    tracker.update(state1)

    stats = tracker.profiles_raw[4]
    assert stats["hands_played"] == 0

    # Simulate a new hand by street going backwards
    # We remove the action so it doesn't immediately trigger VPIP for the new hand
    player2 = PlayerState(seat=4, is_hero=False, is_active=True)
    state2 = TableState(
        is_hand_in_progress=True,
        street=Street.PREFLOP,
        pot=15.0,
        players=[player2]
    )
    tracker.update(state2)

    stats = tracker.profiles_raw[4]
    assert stats["hands_played"] == 1
    assert stats["vpip_this_hand"] is False
    assert stats["pfr_this_hand"] is False

def test_sizing_tells(tracker):
    player = PlayerState(seat=5, is_hero=False, is_active=True, last_action=ActionType.RAISE, bet=25.0)
    state = TableState(
        is_hand_in_progress=True,
        street=Street.PREFLOP,
        pot=30.0, # pot > big_blind * 2 (big_blind defaults to 1.0)
        big_blind=1.0,
        players=[player]
    )
    tracker.update(state)

    stats = tracker.profiles_raw[5]
    assert stats["three_bet_opps"] == 1
    assert stats["three_bets"] == 1
    assert player.profile.three_bet_pct == 1.0
