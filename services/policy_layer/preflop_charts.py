# ruff: noqa: E501
from libs.common.schemas import ActionType


def expand_range(range_str: str) -> set[str]:
    """
    Expands a standard poker range string into a set of hand codes.
    Example: "22+, A2s+, KTs+, AJo+" -> {"AA", "KK", ..., "AKs", "AQs", ..., "AKo", "AQo", ...}
    """
    if not range_str:
        return set()

    ranks = '23456789TJQKA'
    rank_idx = {r: i for i, r in enumerate(ranks)}

    result = set()
    for token in range_str.replace(" ", "").split(','):
        if not token:
            continue
        if token.endswith('+'):
            base = token[:-1]
            if len(base) == 2:  # e.g. "22+"
                r = base[0]
                idx = rank_idx[r]
                for i in range(idx, len(ranks)):
                    result.add(ranks[i] + ranks[i])
            elif len(base) == 3: # e.g. "ATs+", "AJo+"
                r1 = base[0]
                r2 = base[1]
                suit = base[2]
                idx1 = rank_idx[r1]
                idx2 = rank_idx[r2]
                for i in range(idx2, idx1):
                    result.add(r1 + ranks[i] + suit)
        elif '-' in token:
            parts = token.split('-')
            base1, base2 = parts[0], parts[1]
            if len(base1) == 2: # "22-55"
                r1 = base1[0]
                r2 = base2[0]
                for i in range(rank_idx[r1], rank_idx[r2] + 1):
                    result.add(ranks[i] + ranks[i])
            elif len(base1) == 3: # "A2s-A5s"
                r = base1[0]
                r1 = base1[1]
                r2 = base2[1]
                suit = base1[2]
                for i in range(rank_idx[r1], rank_idx[r2] + 1):
                    result.add(r + ranks[i] + suit)
        else:
            result.add(token)
    return result

# Hardcoded GTO ranges for 6-max and 9-max for each position (UTG, MP, CO, BTN, SB, BB)
# Format: dict[position][action] = set of hands.

_CHARTS_6MAX_STR = {
    "UTG": {
        ActionType.RAISE: "77+, A2s+, K9s+, Q9s+, J9s+, T9s, AJo+, KQo",
        ActionType.CALL: "55-66, AQs, KQs, JTs, T9s",
    },
    "MP": {
        ActionType.RAISE: "55+, A2s+, K8s+, Q9s+, J9s+, T9s, 98s, 87s, ATo+, KQo, KJo",
        ActionType.CALL: "22-44, AJs, KTs, QTs, JTs, T9s, 98s",
    },
    "CO": {
        ActionType.RAISE: "22+,  # noqa: E501 A2s+, K2s+, Q5s+, J7s+, T7s+, 97s+, 87s, 76s, 65s, 54s, A8o+, KTo+, QTo+, JTo",
        ActionType.CALL: "A2s-A9s, K9s, Q9s, J9s, T9s, 98s, 87s",
    },
    "BTN": {
        ActionType.RAISE: "22+,  # noqa: E501 A2s+, K2s+, Q2s+, J2s+, T5s+, 95s+, 85s+, 74s+, 64s+, 53s+, 43s, A2o+, K8o+, Q9o+, J9o+, T9o, 98o",
        ActionType.CALL: "A2s-A7s, K2s-K8s, Q2s-Q8s, J7s, T8s, 98s, 87s",
    },
    "SB": {
        ActionType.RAISE: "22+,  # noqa: E501 A2s+, K2s+, Q2s+, J4s+, T6s+, 96s+, 86s+, 75s+, 65s, 54s, A2o+, K8o+, Q9o+, J9o+, T9o",
        ActionType.CALL: "A2s-A9s, K2s-K8s, Q2s-Q8s",
    },
    "BB": {
        ActionType.RAISE: "88+, A2s+, KTs+, QTs+, JTs, AJo+, KQo",
        ActionType.CALL: "22-77,  # noqa: E501 A2s-A9s, K2s-K9s, Q2s-Q9s, J2s-J9s, T2s-T9s, 92s-98s, 84s-87s, 74s-76s, 64s-65s, 53s-54s, 43s, A2o-ATo, K2o-KJo, Q5o-QJo, J7o-JTo, T7o-T9o, 97o-98o, 87o",
    }
}

_CHARTS_9MAX_STR = {
    "UTG": {
        ActionType.RAISE: "88+, A8s+, KQs, AJo+, KQo",
        ActionType.CALL: "55-77, A2s-A7s, KJs, QJs, JTs, T9s",
    },
    "MP": {
        ActionType.RAISE: "77+, A2s+, K9s+, Q9s+, J9s+, T9s, ATo+, KQo",
        ActionType.CALL: "22-66, AQs, KQs, JTs, T9s, 98s",
    },
    "CO": {
        ActionType.RAISE: "55+, A2s+, K8s+, Q9s+, J9s+, T9s, 98s, 87s, ATo+, KJo+, QJo",
        ActionType.CALL: "22-44, A2s-A9s, K9s, Q9s, JTs, T9s, 98s, 87s",
    },
    "BTN": {
        ActionType.RAISE: "22+,  # noqa: E501 A2s+, K2s+, Q5s+, J7s+, T7s+, 97s+, 87s, 76s, 65s, 54s, A8o+, KTo+, QTo+, JTo",
        ActionType.CALL: "A2s-A9s, K2s-K8s, Q2s-Q8s, J9s, T8s, 98s, 87s",
    },
    "SB": {
        ActionType.RAISE: "22+,  # noqa: E501 A2s+, K2s+, Q2s+, J4s+, T6s+, 96s+, 86s+, 75s+, 65s, 54s, A2o+, K8o+, Q9o+, J9o+, T9o",
        ActionType.CALL: "A2s-A9s, K2s-K8s, Q2s-Q8s",
    },
    "BB": {
        ActionType.RAISE: "88+, A2s+, KTs+, QTs+, JTs, AJo+, KQo",
        ActionType.CALL: "22-77,  # noqa: E501 A2s-A9s, K2s-K9s, Q2s-Q9s, J2s-J9s, T2s-T9s, 92s-98s, 84s-87s, 74s-76s, 64s-65s, 53s-54s, 43s, A2o-ATo, K2o-KJo, Q5o-QJo, J7o-JTo, T7o-T9o, 97o-98o, 87o",
    }
}

CHARTS_6MAX: dict[str, dict[ActionType, set[str]]] = {}
for pos, actions in _CHARTS_6MAX_STR.items():
    CHARTS_6MAX[pos] = {}
    for action, rng in actions.items():
        CHARTS_6MAX[pos][action] = expand_range(rng)

CHARTS_9MAX: dict[str, dict[ActionType, set[str]]] = {}
for pos, actions in _CHARTS_9MAX_STR.items():
    CHARTS_9MAX[pos] = {}
    for action, rng in actions.items():
        CHARTS_9MAX[pos][action] = expand_range(rng)

def get_preflop_action(
    table_size: int,
    position: str,
    hand: str
) -> ActionType | None:
    """
    Look up the hand in the preflop charts for the given position.
    Returns the recommended ActionType or None.
    Hand should be in canonical format, e.g., 'AKs', 'AA', '76o'.
    """
    charts = CHARTS_6MAX if table_size <= 6 else CHARTS_9MAX
    pos_charts = charts.get(position, {})

    # Priority: Raise/3Bet -> Call
    if hand in pos_charts.get(ActionType.RAISE, set()):
        return ActionType.RAISE
    if hand in pos_charts.get(ActionType.CALL, set()):
        return ActionType.CALL

    return None
