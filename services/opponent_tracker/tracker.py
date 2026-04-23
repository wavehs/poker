from typing import Dict, Any, List, Set
from collections import defaultdict
from pydantic import BaseModel, Field

from libs.common.schemas import ActionType, Street, OpponentProfile

class OpponentData(BaseModel):
    seat_id: int
    hands_played: Set[str] = Field(default_factory=set)
    vpip_opportunities: int = 0
    vpip_actions: int = 0
    pfr_opportunities: int = 0
    pfr_actions: int = 0
    aggressive_actions: int = 0
    passive_actions: int = 0
    cbet_faced: int = 0
    cbet_folded: int = 0
    bet_sizes_by_street: Dict[Street, List[float]] = Field(default_factory=lambda: defaultdict(list))


class OpponentTracker:
    def __init__(self) -> None:
        self._data: Dict[int, OpponentData] = {}
        # Keep track of the last action in the hand per street to infer fold to cbet
        # A bit complex to do full fold_to_cbet tracking without full hand history context,
        # but we can try to estimate or simply add faced_cbet parameter or logic.
        # Let's keep it simple for now.

    def _get_or_create(self, seat_id: int) -> OpponentData:
        if seat_id not in self._data:
            self._data[seat_id] = OpponentData(seat_id=seat_id)
        return self._data[seat_id]

    def record_action(
        self,
        seat_id: int,
        hand_id: str,
        action: ActionType,
        sizing_ratio: float,
        time_to_act_ms: float,
        street: Street,
        position: str,
        won_showdown: bool
    ) -> None:
        data = self._get_or_create(seat_id)

        # Track hand
        data.hands_played.add(hand_id)

        # Action classification
        is_aggressive = action in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN)
        is_passive = action == ActionType.CALL

        if is_aggressive:
            data.aggressive_actions += 1
            if sizing_ratio > 0:
                data.bet_sizes_by_street[street].append(sizing_ratio)
        elif is_passive:
            data.passive_actions += 1

        # VPIP & PFR (only on PREFLOP for simplicity in this tracker)
        if street == Street.PREFLOP:
            # Simplistic: every action is an opportunity preflop
            data.vpip_opportunities += 1
            if action in (ActionType.CALL, ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
                data.vpip_actions += 1

            # PFR opportunity: any action preflop could be a raise opportunity
            data.pfr_opportunities += 1
            if action in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
                data.pfr_actions += 1

        # Fold to CBet logic (Flop only for simplicity, assuming any fold on flop is fold to cbet if we don't have full state)
        # Without full state, we'll just track if they folded on the flop. We might need a heuristic.
        # Let's say if they fold on flop, we count it as a cbet_faced and cbet_folded.
        # If they call/raise on flop, we count as cbet_faced.
        # This is an approximation since we don't know who the preflop aggressor is.
        if street == Street.FLOP:
            if action == ActionType.FOLD:
                data.cbet_faced += 1
                data.cbet_folded += 1
            elif action in (ActionType.CALL, ActionType.RAISE, ActionType.ALL_IN):
                data.cbet_faced += 1

    def get_profile(self, seat_id: int) -> OpponentProfile:
        if seat_id not in self._data:
            return OpponentProfile(seat_id=seat_id)

        data = self._data[seat_id]

        vpip = data.vpip_actions / data.vpip_opportunities if data.vpip_opportunities > 0 else 0.0
        pfr = data.pfr_actions / data.pfr_opportunities if data.pfr_opportunities > 0 else 0.0

        af = 0.0
        if data.passive_actions > 0:
            af = data.aggressive_actions / data.passive_actions
        elif data.aggressive_actions > 0:
            af = float('inf') # Or maybe just aggressive_actions? We'll use aggressive_actions

        fold_to_cbet = data.cbet_folded / data.cbet_faced if data.cbet_faced > 0 else 0.0

        sizing_tells = {}
        for s, sizes in data.bet_sizes_by_street.items():
            if sizes:
                sizing_tells[s] = sum(sizes) / len(sizes)

        return OpponentProfile(
            seat_id=seat_id,
            hands_seen=len(data.hands_played),
            vpip=vpip,
            pfr=pfr,
            af=af,
            fold_to_cbet=fold_to_cbet,
            sizing_tells=sizing_tells,
            sample_size=sum([data.vpip_opportunities, data.cbet_faced, data.aggressive_actions, data.passive_actions])
        )

    def reset_session(self) -> None:
        self._data.clear()
