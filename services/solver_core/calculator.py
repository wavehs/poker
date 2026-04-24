def calculate_spr(effective_stack: float, pot_size: float) -> float:
    """
    Calculate Stack-to-Pot Ratio (SPR).
    spr = effective_stack / pot_size
    """
    if pot_size <= 0:
        return float('inf')
    return effective_stack / pot_size

def get_spr_advice(spr: float) -> str:
    """
    Get SPR-based recommendations based on standard thresholds (1/4/10/20).
    """
    if spr < 1:
        return "SPR < 1: Автоматический all-in или fold. Вы привязаны к банку."
    elif spr < 4:
        return "SPR 1-4: Низкий SPR. Готовность играть на стек с топ-парой или сильным дро."
    elif spr < 10:
        return "SPR 4-10: Средний SPR. Осторожно с одной парой, для игры на стек нужны две пары+."
    elif spr < 20:
        return "SPR 10-20: Высокий SPR. Фолд с топ-парой при сильной агрессии, требуются натсы."
    else:
        return "SPR 20+: Очень высокий SPR. Игра на стек только с абсолютными натсами, высокое фолд-эквити."
