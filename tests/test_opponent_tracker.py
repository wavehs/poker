"""Tests for Opponent Tracker."""

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
        pot=30.0,
        big_blind=1.0,
        players=[player]
    )
    tracker.update(state)

    stats = tracker.profiles_raw[5]
    assert stats["three_bet_opps"] == 1
    assert stats["three_bets"] == 1
    assert player.profile.three_bet_pct == 1.0


class TestOpponentTracker:
    def test_get_exploits_cbet_always(self):
        tracker = OpponentTracker()
        tracker.profiles_raw[1] = {
            "hands_played": 50,
            "vpip_hands": 10,
            "pfr_hands": 5,
            "aggr_actions": 5,
            "pass_actions": 20,
            "three_bet_opps": 10,
            "three_bets": 1,
            "faced_cbet": 10,
            "folded_to_cbet": 8,
            "vpip_this_hand": False,
            "pfr_this_hand": False,
        }

        exploits = tracker.get_exploits(1)
        assert len(exploits) >= 1
        cbet_exploit = next((e for e in exploits if e["exploit"] == "cbet_always"), None)
        assert cbet_exploit is not None
        assert cbet_exploit["confidence"] == 0.8

    def test_get_exploits_value_bet_thin(self):
        tracker = OpponentTracker()
        tracker.profiles_raw[2] = {
            "hands_played": 50,
            "vpip_hands": 30,
            "pfr_hands": 5,
            "aggr_actions": 10,
            "pass_actions": 40,
            "three_bet_opps": 10,
            "three_bets": 1,
            "faced_cbet": 10,
            "folded_to_cbet": 3,
            "vpip_this_hand": False,
            "pfr_this_hand": False,
        }

        exploits = tracker.get_exploits(2)
        assert len(exploits) >= 1
        value_exploit = next((e for e in exploits if e["exploit"] == "value_bet_thin"), None)
        assert value_exploit is not None
        assert value_exploit["confidence"] == 0.75

    def test_get_exploits_steal_blinds(self):
        tracker = OpponentTracker()
        tracker.profiles_raw[3] = {
            "hands_played": 50,
            "vpip_hands": 5,
            "pfr_hands": 2,
            "aggr_actions": 5,
            "pass_actions": 5,
            "three_bet_opps": 10,
            "three_bets": 1,
            "faced_cbet": 10,
            "folded_to_cbet": 3,
            "vpip_this_hand": False,
            "pfr_this_hand": False,
        }

        exploits = tracker.get_exploits(3)
        assert len(exploits) >= 1
        steal_exploit = next((e for e in exploits if e["exploit"] == "steal_blinds"), None)
        assert steal_exploit is not None
        assert steal_exploit["confidence"] == 0.5

    def test_get_exploits_no_data(self):
        tracker = OpponentTracker()
        exploits = tracker.get_exploits(99)
        assert exploits == []

    def test_get_exploits_insufficient_sample(self):
        tracker = OpponentTracker()
        tracker.profiles_raw[4] = {
            "hands_played": 4,
            "vpip_hands": 4,
            "pfr_hands": 4,
            "aggr_actions": 4,
            "pass_actions": 0,
            "three_bet_opps": 0,
            "three_bets": 0,
            "faced_cbet": 4,
            "folded_to_cbet": 4,
            "vpip_this_hand": False,
            "pfr_this_hand": False,
        }

        exploits = tracker.get_exploits(4)
        assert exploits == []