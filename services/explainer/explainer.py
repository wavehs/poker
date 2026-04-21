"""
Explainer — Human-readable explanation generator.

Transforms structured recommendations into concise,
actionable explanations. No LLM — template-based only.
"""

from __future__ import annotations

from libs.common.schemas import (
    ActionType,
    Recommendation,
    Street,
    TableState,
)


# ─── Street names ────────────────────────────────────────────────────────────

STREET_NAMES = {
    Street.PREFLOP: "Префлоп",
    Street.FLOP: "Флоп",
    Street.TURN: "Тёрн",
    Street.RIVER: "Ривер",
    Street.SHOWDOWN: "Шоудаун",
    Street.UNKNOWN: "—",
}

ACTION_NAMES = {
    ActionType.FOLD: "Фолд",
    ActionType.CHECK: "Чек",
    ActionType.CALL: "Колл",
    ActionType.BET: "Бет",
    ActionType.RAISE: "Рейз",
    ActionType.ALL_IN: "Олл-ин",
    ActionType.UNCERTAIN: "Неизвестно",
}


class Explainer:
    """
    Generates human-readable explanations for poker recommendations.
    
    Uses deterministic templates. No LLM calls.
    """

    def explain(
        self,
        recommendation: Recommendation,
        state: TableState,
    ) -> str:
        """
        Generate a full explanation string.
        
        Args:
            recommendation: The structured recommendation.
            state: Current table state.
            
        Returns:
            Human-readable explanation string.
        """
        if recommendation.is_uncertain:
            return self._explain_uncertain(recommendation)

        parts: list[str] = []

        # Street
        street = STREET_NAMES.get(recommendation.street, "—")
        parts.append(f"📍 {street}")

        # Hero cards
        hero = state.hero
        if hero and hero.hole_cards:
            cards_str = " ".join(c.code for c in hero.hole_cards)
            parts.append(f"🃏 Рука: {cards_str}")

        # Board
        if state.community_cards:
            board_str = " ".join(c.code for c in state.community_cards)
            parts.append(f"🂠 Борд: {board_str}")

        # Core metrics
        parts.append(f"📊 Эквити: {recommendation.equity:.0%}")
        parts.append(f"💪 Сила руки: {recommendation.hand_strength:.0%}")

        if recommendation.pot_odds > 0:
            parts.append(f"💰 Пот-оддсы: {recommendation.pot_odds:.0%}")

        parts.append(f"📐 SPR: {recommendation.spr:.1f}")
        parts.append(f"📏 Эфф. стек: {recommendation.effective_stack_bb:.0f}BB")

        # Best action
        best = recommendation.best_action
        action_name = ACTION_NAMES.get(best.action_type, "—")
        if best.amount > 0:
            parts.append(f"\n✅ Рекомендация: {action_name} {best.amount:.0f}")
        else:
            parts.append(f"\n✅ Рекомендация: {action_name}")

        # Score
        parts.append(f"⭐ Скор: {best.score:.2f}")

        if best.ev != 0:
            parts.append(f"📈 EV: {best.ev:+.1f}")

        # All actions summary
        if len(recommendation.all_actions) > 1:
            parts.append("\n🎯 Все варианты:")
            for act in sorted(recommendation.all_actions, key=lambda a: a.score, reverse=True):
                name = ACTION_NAMES.get(act.action_type, "?")
                amt = f" {act.amount:.0f}" if act.amount > 0 else ""
                parts.append(f"  • {name}{amt} (скор: {act.score:.2f}, EV: {act.ev:+.1f})")

        # Confidence warning
        if recommendation.confidence.is_dangerous:
            parts.append("\n⚠️ ВНИМАНИЕ: Низкая уверенность! Рекомендация может быть неточной.")
            parts.append(
                f"  Vision: {recommendation.confidence.vision_confidence:.0%} | "
                f"OCR: {recommendation.confidence.ocr_confidence:.0%} | "
                f"State: {recommendation.confidence.state_confidence:.0%}"
            )

        # Existing explanation from policy
        if recommendation.explanation:
            parts.append(f"\n💡 {recommendation.explanation}")

        return "\n".join(parts)

    def explain_short(self, recommendation: Recommendation) -> str:
        """
        Generate a compact one-line explanation.
        
        Format: ACTION | equity% | EV±X | confidence%
        """
        if recommendation.is_uncertain:
            return "⚠️ Неизвестно — низкая уверенность"

        best = recommendation.best_action
        action_name = ACTION_NAMES.get(best.action_type, "?")
        amt = f" {best.amount:.0f}" if best.amount > 0 else ""

        conf = recommendation.confidence.overall
        danger = " ⚠️" if recommendation.confidence.is_dangerous else ""

        return (
            f"{action_name}{amt} | "
            f"Equity {recommendation.equity:.0%} | "
            f"EV {best.ev:+.1f} | "
            f"Conf {conf:.0%}{danger}"
        )

    def _explain_uncertain(self, recommendation: Recommendation) -> str:
        """Explain why the recommendation is uncertain."""
        parts = [
            "⚠️ СОСТОЯНИЕ НЕОПРЕДЕЛЁННО",
            "",
            recommendation.explanation or "Недостаточно данных для рекомендации.",
            "",
            "Уверенность:",
            f"  Vision: {recommendation.confidence.vision_confidence:.0%}",
            f"  OCR: {recommendation.confidence.ocr_confidence:.0%}",
            f"  State: {recommendation.confidence.state_confidence:.0%}",
            "",
            "Действуйте по своему усмотрению.",
        ]
        return "\n".join(parts)
