"""Tests for Opponent Tracker."""

from libs.common.schemas import ActionType, Street, TableState, PlayerState
from services.opponent_tracker.tracker import OpponentTracker

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
        # fold_cbet is 0.8, sample size >= 10, modifier is 1.0, conf is 0.8
        assert len(exploits) >= 1
        cbet_exploit = next((e for e in exploits if e["exploit"] == "cbet_always"), None)
        assert cbet_exploit is not None
        assert cbet_exploit["confidence"] == 0.8

    def test_get_exploits_value_bet_thin(self):
        tracker = OpponentTracker()
        tracker.profiles_raw[2] = {
            "hands_played": 50,
            "vpip_hands": 30, # VPIP = 0.6
            "pfr_hands": 5,
            "aggr_actions": 10,
            "pass_actions": 40, # AF = 0.25
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
        # conf = min(0.95, vpip * (1.5 - af)) = min(0.95, 0.6 * (1.5 - 0.25)) = min(0.95, 0.6 * 1.25) = min(0.95, 0.75) = 0.75
        assert value_exploit["confidence"] == 0.75

    def test_get_exploits_steal_blinds(self):
        tracker = OpponentTracker()
        tracker.profiles_raw[3] = {
            "hands_played": 50,
            "vpip_hands": 5, # VPIP = 0.1
            "pfr_hands": 2,
            "aggr_actions": 5,
            "pass_actions": 5, # AF = 1.0
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
        # conf = min(0.95, 1.0 - (vpip * 5)) = min(0.95, 1.0 - (0.1 * 5)) = 0.5
        assert steal_exploit["confidence"] == 0.5

    def test_get_exploits_no_data(self):
        tracker = OpponentTracker()
        exploits = tracker.get_exploits(99)
        assert exploits == []

    def test_get_exploits_insufficient_sample(self):
        tracker = OpponentTracker()
        tracker.profiles_raw[4] = {
            "hands_played": 4, # Less than 5 hands
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
