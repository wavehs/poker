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

        # Hand-level state (reset on each new hand)
        # Seat of the most recent preflop aggressor (open-raiser).
        self.preflop_aggressor_seat: int | None = None

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
        """Detect whether a new hand started since the last update.

        Heuristics:
          1. The street regresses to PREFLOP from a postflop street.
          2. The community-card count drops to zero from non-zero (board reset).
          3. The pot resets below the big blind from a non-trivial previous pot.
        """
        street_order = {
            Street.UNKNOWN: -1,
            Street.PREFLOP: 0,
            Street.FLOP: 1,
            Street.TURN: 2,
            Street.RIVER: 3,
            Street.SHOWDOWN: 4,
        }
        current = street_order.get(state.street, -1)
        last = street_order.get(self.last_street, -1)

        if last >= 1 and current == 0:
            return True

        if self.last_community_cards_count > 0 and len(state.community_cards) == 0:
            return True

        bb = max(state.big_blind, 1.0)
        if self.last_pot > bb * 3 and state.pot <= bb:
            return True

        return False

    def _reset_hand_state(self, state: TableState) -> None:
        """Reset internal trackers for a new hand.

        Only seats that are currently active at the table are credited with a
        new hand played. Seats that have left the table no longer accumulate
        hands they didn't actually play.
        """
        self.last_observed_action.clear()
        active_seats = {p.seat for p in state.players if p.is_active and not p.is_hero}
        for seat, stats in self.profiles_raw.items():
            if seat in active_seats:
                stats["hands_played"] += 1
            stats["vpip_this_hand"] = False
            stats["pfr_this_hand"] = False
        self.preflop_aggressor_seat = None

    def update(self, state: TableState) -> None:
        """
        Process the new state, update raw counters, and attach OpponentProfile to players.
        """
        if not state.is_hand_in_progress and state.pot == 0:
            return  # Wait until a hand actually begins

        if self._detect_new_hand(state):
            self._reset_hand_state(state)

        # Ensure every currently-active villain has a profile entry so the
        # per-hand reset above doesn't miss new arrivals.
        for p in state.players:
            if not p.is_hero and p.seat is not None:
                self._init_seat(p.seat)

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
                    voluntary = (
                        ActionType.CALL,
                        ActionType.BET,
                        ActionType.RAISE,
                        ActionType.ALL_IN,
                    )
                    if action in voluntary and not stats["vpip_this_hand"]:
                        stats["vpip_this_hand"] = True
                        stats["vpip_hands"] += 1

                    if (
                        action in (ActionType.RAISE, ActionType.ALL_IN)
                        and not stats["pfr_this_hand"]
                    ):
                        stats["pfr_this_hand"] = True
                        stats["pfr_hands"] += 1

                    # 3-bet: a different seat re-raises after a prior open.
                    is_aggression = action in (ActionType.RAISE, ActionType.ALL_IN)
                    prior_aggressor = self.preflop_aggressor_seat
                    facing_open = prior_aggressor is not None and prior_aggressor != p.seat
                    if facing_open:
                        stats["three_bet_opps"] += 1
                        if is_aggression:
                            stats["three_bets"] += 1
                    if is_aggression:
                        # This seat is now the most recent preflop aggressor.
                        self.preflop_aggressor_seat = p.seat

                # C-Bet: count only when this seat was NOT the preflop
                # aggressor and is now facing a flop bet from someone else.
                if (
                    state.street == Street.FLOP
                    and self.preflop_aggressor_seat is not None
                    and self.preflop_aggressor_seat != p.seat
                    and p.bet > 0
                ):
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

    def get_exploits(self, seat_id: int) -> list[dict[str, str | float]]:
        """
        Analyze accumulated statistics and return a list of specific exploits with confidence.
        Returns a list of dicts like: [{"exploit": "cbet_always", "confidence": 0.82}]
        """
        if seat_id not in self.profiles_raw:
            return []

        stats = self.profiles_raw[seat_id]
        hands = int(stats.get("hands_played", 0))

        # We need a minimal sample size to be confident in any exploit
        if hands < 5:
            return []

        exploits: list[dict[str, str | float]] = []

        # 1. cbet_always: Exploit players who fold too much to c-bets
        faced_cbet = int(stats.get("faced_cbet", 0))
        if faced_cbet >= 3:
            fold_cbet = float(stats["folded_to_cbet"]) / faced_cbet
            if fold_cbet >= 0.60:
                # Confidence scales with sample size, maxing out at fold_cbet value
                sample_confidence_modifier = min(1.0, faced_cbet / 10.0)
                confidence = round(fold_cbet * sample_confidence_modifier, 2)
                if confidence >= 0.5:
                    exploits.append({"exploit": "cbet_always", "confidence": confidence})

        # 2. value_bet_thin: Exploit calling stations (high VPIP, low AF)
        vpip = float(stats["vpip_hands"]) / hands
        aggr = float(stats["aggr_actions"])
        pass_act = float(stats["pass_actions"])
        af = aggr / pass_act if pass_act > 0 else (aggr if aggr > 0 else 0.0)

        if hands >= 10 and vpip > 0.40 and af < 1.0:
            confidence = round(min(0.95, vpip * (1.5 - af)), 2)
            if confidence >= 0.5:
                exploits.append({"exploit": "value_bet_thin", "confidence": confidence})

        # 3. steal_blinds: Exploit tight players (low VPIP)
        if hands >= 10 and vpip < 0.15:
            # lower VPIP means higher confidence in steal
            confidence = round(min(0.95, 1.0 - (vpip * 5)), 2)
            if confidence >= 0.5:
                exploits.append({"exploit": "steal_blinds", "confidence": confidence})

        return exploits
