"""
Opponent Tracker Service

Collects per-session statistics by seat_id and provides an OpponentProfile
that can be injected into the policy layer.
"""

from libs.common.schemas import ActionType, OpponentProfile, Street, TableState

class OpponentTracker:
    def __init__(self) -> None:
        # Dictionary storing raw metrics for each seat_id
        # In-memory storage without DB as requested
        self.profiles_raw: dict[int, dict[str, int | bool]] = {}

        # State tracking to identify new hands and unique actions
        self.last_pot = 0.0
        self.last_street = Street.UNKNOWN
        self.last_community_cards_count = 0
        # Now keyed by (seat_id, street) to allow same actions on different streets
        self.last_observed_action: dict[tuple[int, Street], ActionType | None] = {}

    def _init_seat(self, seat: int) -> None:
        if seat not in self.profiles_raw:
            self.profiles_raw[seat] = {
                "hands_played": 0,
                "vpip_hands": 0,
                "pfr_hands": 0,
                "aggr_actions": 0,
                "pass_actions": 0,
                "three_bet_opps": 0,
                "three_bets": 0,
                "faced_cbet": 0,
                "folded_to_cbet": 0,
                # internal hand state
                "vpip_this_hand": False,
                "pfr_this_hand": False,
            }

    def _detect_new_hand(self, state: TableState) -> bool:
        """Detect if a new hand has started based on board state changes."""
        is_new_hand = False

        # Heuristics for a new hand:
        # 1. Street goes backwards (e.g., Flop -> Preflop)
        street_order = {
            Street.UNKNOWN: -1,
            Street.PREFLOP: 0,
            Street.FLOP: 1,
            Street.TURN: 2,
            Street.RIVER: 3,
            Street.SHOWDOWN: 4
        }

        current_street_val = street_order.get(state.street, -1)
        last_street_val = street_order.get(self.last_street, -1)

        if current_street_val < last_street_val and current_street_val == 0:
            is_new_hand = True

        # 2. Community cards decrease
        if len(state.community_cards) < self.last_community_cards_count:
            is_new_hand = True

        # 3. Pot drops significantly
        if state.pot < self.last_pot and state.pot < 10.0 and self.last_pot > 0:
            is_new_hand = True

        return is_new_hand

    def _reset_hand_state(self) -> None:
        """Reset internal trackers for a new hand."""
        self.last_observed_action.clear()
        for seat, stats in self.profiles_raw.items():
            stats["hands_played"] += 1
            stats["vpip_this_hand"] = False
            stats["pfr_this_hand"] = False

    def update(self, state: TableState) -> None:
        """
        Process the new state, update raw counters, and attach OpponentProfile to players.
        """
        if not state.is_hand_in_progress and state.pot == 0:
            return  # Wait until a hand actually begins

        if self._detect_new_hand(state):
            self._reset_hand_state()

        self.last_pot = state.pot
        self.last_street = state.street
        self.last_community_cards_count = len(state.community_cards)

        # Process actions
        for p in state.players:
            if p.is_hero or not p.is_active:
                continue

            self._init_seat(p.seat)
            stats = self.profiles_raw[p.seat]

            # Action changed or just processed first time this street
            last_action = self.last_observed_action.get((p.seat, state.street))
            if p.last_action and p.last_action != last_action:
                self.last_observed_action[(p.seat, state.street)] = p.last_action
                action = p.last_action

                # Update stats
                if state.street == Street.PREFLOP:
                    if action in (ActionType.CALL, ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
                        if not stats["vpip_this_hand"]:
                            stats["vpip_this_hand"] = True
                            stats["vpip_hands"] += 1

                    if action in (ActionType.RAISE, ActionType.ALL_IN):
                        if not stats["pfr_this_hand"]:
                            stats["pfr_this_hand"] = True
                            stats["pfr_hands"] += 1

                    # Basic 3-bet logic: If there was a previous raise by someone else
                    # and this player raises again preflop.
                    # We will simplify: If the pot is big enough preflop, and they raise,
                    # we'll approximate it as a 3-bet opportunity.
                    if p.bet > 0 and state.pot > state.big_blind * 2:
                        stats["three_bet_opps"] += 1
                        if action in (ActionType.RAISE, ActionType.ALL_IN):
                            stats["three_bets"] += 1

                # C-Bet basic heuristics: if flop/turn, someone bets
                if state.street in (Street.FLOP, Street.TURN, Street.RIVER) and state.pot > 0:
                    # If hero is facing a bet, it's a c-bet scenario approx
                    stats["faced_cbet"] += 1
                    if action == ActionType.FOLD:
                        stats["folded_to_cbet"] += 1

                # Aggression factor counters
                if action in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
                    stats["aggr_actions"] += 1
                elif action == ActionType.CALL:
                    stats["pass_actions"] += 1

        # Compute and attach profiles
        for p in state.players:
            if not p.is_hero and p.seat in self.profiles_raw:
                p.profile = self._compute_profile(self.profiles_raw[p.seat])

    def _compute_profile(self, stats: dict[str, int | bool]) -> OpponentProfile:
        """Compute the final percentages for the OpponentProfile."""
        hands = int(stats["hands_played"]) or 1  # prevent div 0

        vpip = float(stats["vpip_hands"]) / hands
        pfr = float(stats["pfr_hands"]) / hands

        aggr = float(stats["aggr_actions"])
        pass_act = float(stats["pass_actions"])
        af = aggr / pass_act if pass_act > 0 else (aggr if aggr > 0 else 0.0)

        # Basic approximations for complex stats if full hand history isn't perfectly parsed
        three_bet_pct = float(stats["three_bets"]) / max(1, int(stats["three_bet_opps"]))
        fold_cbet = float(stats["folded_to_cbet"]) / max(1, int(stats["faced_cbet"]))

        return OpponentProfile(
            vpip=min(1.0, vpip),
            pfr=min(1.0, pfr),
            af=af,
            three_bet_pct=min(1.0, three_bet_pct),
            fold_to_cbet_pct=min(1.0, fold_cbet),
            hands_played=int(stats["hands_played"])
        )
