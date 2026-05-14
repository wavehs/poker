# ruff: noqa: E501
import logging

from libs.common.schemas import ActionType

logger = logging.getLogger(__name__)

_RANKS = "23456789TJQKA"
_RANK_IDX = {r: i for i, r in enumerate(_RANKS)}
_SUITS = {"s", "o"}


def _is_valid_hand_code(code: str) -> bool:
    """Validate a single hand code like 'AA', 'AKs', or 'AKo'."""
    if len(code) == 2:
        return code[0] in _RANK_IDX and code[0] == code[1]
    if len(code) == 3:
        return (
            code[0] in _RANK_IDX
            and code[1] in _RANK_IDX
            and code[2] in _SUITS
            and code[0] != code[1]
        )
    return False


def expand_range(range_str: str) -> set[str]:
    """
    Expands a standard poker range string into a set of hand codes.
    Example: "22+, A2s+, KTs+, AJo+" -> {"AA", "KK", ..., "AKs", "AQs", ..., "AKo", "AQo", ...}

    Tokens that don't match the expected formats ("XX", "XYs/o", "XX+", "XYs/o+",
    "XX-XX", "XYs-XYs") are logged as warnings and ignored. This guards against
    silently-dropped tokens caused by typos or accidentally embedded comments.
    """
    if not range_str:
        return set()

    result: set[str] = set()
    for raw in range_str.split(","):
        token = raw.strip().replace(" ", "")
        if not token:
            continue

        if token.endswith("+"):
            base = token[:-1]
            if len(base) == 2 and base[0] == base[1] and base[0] in _RANK_IDX:
                idx = _RANK_IDX[base[0]]
                for i in range(idx, len(_RANKS)):
                    result.add(_RANKS[i] + _RANKS[i])
                continue
            if (
                len(base) == 3
                and base[0] in _RANK_IDX
                and base[1] in _RANK_IDX
                and base[2] in _SUITS
                and _RANK_IDX[base[0]] > _RANK_IDX[base[1]]
            ):
                idx1 = _RANK_IDX[base[0]]
                idx2 = _RANK_IDX[base[1]]
                for i in range(idx2, idx1):
                    result.add(base[0] + _RANKS[i] + base[2])
                continue
            logger.warning("expand_range: ignoring malformed '+' token %r", token)
            continue

        if "-" in token:
            parts = token.split("-")
            if len(parts) != 2:
                logger.warning("expand_range: ignoring malformed '-' token %r", token)
                continue
            base1, base2 = parts
            if (
                len(base1) == 2
                and base1[0] == base1[1]
                and len(base2) == 2
                and base2[0] == base2[1]
                and base1[0] in _RANK_IDX
                and base2[0] in _RANK_IDX
            ):
                lo, hi = sorted((_RANK_IDX[base1[0]], _RANK_IDX[base2[0]]))
                for i in range(lo, hi + 1):
                    result.add(_RANKS[i] + _RANKS[i])
                continue
            if (
                len(base1) == 3
                and len(base2) == 3
                and base1[0] == base2[0]
                and base1[2] == base2[2]
                and base1[0] in _RANK_IDX
                and base1[1] in _RANK_IDX
                and base2[1] in _RANK_IDX
                and base1[2] in _SUITS
            ):
                lo, hi = sorted((_RANK_IDX[base1[1]], _RANK_IDX[base2[1]]))
                for i in range(lo, hi + 1):
                    result.add(base1[0] + _RANKS[i] + base1[2])
                continue
            logger.warning("expand_range: ignoring malformed '-' token %r", token)
            continue

        if _is_valid_hand_code(token):
            result.add(token)
        else:
            logger.warning("expand_range: ignoring unrecognized token %r", token)

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
        ActionType.RAISE: "22+, A2s+, K2s+, Q5s+, J7s+, T7s+, 97s+, 87s, 76s, 65s, 54s, A8o+, KTo+, QTo+, JTo",
        ActionType.CALL: "A2s-A9s, K9s, Q9s, J9s, T9s, 98s, 87s",
    },
    "BTN": {
        ActionType.RAISE: "22+, A2s+, K2s+, Q2s+, J2s+, T5s+, 95s+, 85s+, 74s+, 64s+, 53s+, 43s, A2o+, K8o+, Q9o+, J9o+, T9o, 98o",
        ActionType.CALL: "A2s-A7s, K2s-K8s, Q2s-Q8s, J7s, T8s, 98s, 87s",
    },
    "SB": {
        ActionType.RAISE: "22+, A2s+, K2s+, Q2s+, J4s+, T6s+, 96s+, 86s+, 75s+, 65s, 54s, A2o+, K8o+, Q9o+, J9o+, T9o",
        ActionType.CALL: "A2s-A9s, K2s-K8s, Q2s-Q8s",
    },
    "BB": {
        ActionType.RAISE: "88+, A2s+, KTs+, QTs+, JTs, AJo+, KQo",
        ActionType.CALL: "22-77, A2s-A9s, K2s-K9s, Q2s-Q9s, J2s-J9s, T2s-T9s, 92s-98s, 84s-87s, 74s-76s, 64s-65s, 53s-54s, 43s, A2o-ATo, K2o-KJo, Q5o-QJo, J7o-JTo, T7o-T9o, 97o-98o, 87o",
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
        ActionType.RAISE: "22+, A2s+, K2s+, Q5s+, J7s+, T7s+, 97s+, 87s, 76s, 65s, 54s, A8o+, KTo+, QTo+, JTo",
        ActionType.CALL: "A2s-A9s, K2s-K8s, Q2s-Q8s, J9s, T8s, 98s, 87s",
    },
    "SB": {
        ActionType.RAISE: "22+, A2s+, K2s+, Q2s+, J4s+, T6s+, 96s+, 86s+, 75s+, 65s, 54s, A2o+, K8o+, Q9o+, J9o+, T9o",
        ActionType.CALL: "A2s-A9s, K2s-K8s, Q2s-Q8s",
    },
    "BB": {
        ActionType.RAISE: "88+, A2s+, KTs+, QTs+, JTs, AJo+, KQo",
        ActionType.CALL: "22-77, A2s-A9s, K2s-K9s, Q2s-Q9s, J2s-J9s, T2s-T9s, 92s-98s, 84s-87s, 74s-76s, 64s-65s, 53s-54s, 43s, A2o-ATo, K2o-KJo, Q5o-QJo, J7o-JTo, T7o-T9o, 97o-98o, 87o",
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
