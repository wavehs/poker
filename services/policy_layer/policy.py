"""
Policy Layer — Deterministic action recommendation engine.

Combines equity, pot odds, stack-to-pot ratio, and play style
to produce structured action recommendations.

This is the PRIMARY decision engine. No LLM involved.
"""

from __future__ import annotations

from libs.common.schemas import (
    Action,
    ActionType,
    ConfidenceReport,
    PlayStyle,
    Recommendation,
    Street,
    TableState,
)
from services.solver_core.solver import EquitySolver
from services.policy_layer.range_models import estimate_opponent_range, range_to_cards


class PolicyEngine:
    """
    Deterministic policy engine that produces action recommendations.
    
    Decision logic:
    1. Compute equity via Monte Carlo
    2. Compare equity to pot odds
    3. Factor in SPR and effective stack
    4. Apply play style adjustments
    5. Score all possible actions
    6. Return structured recommendation
    """

    # Play style adjustments to equity thresholds
    STYLE_ADJUSTMENTS = {
        PlayStyle.AGGRESSIVE: -0.05,    # Lower threshold = more aggressive
        PlayStyle.BALANCED: 0.0,
        PlayStyle.CONSERVATIVE: 0.05,   # Higher threshold = more cautious
    }

    # Push/fold threshold in big blinds
    SHORT_STACK_THRESHOLD_BB = 12.0

    def __init__(
        self,
        solver: EquitySolver | None = None,
        play_style: PlayStyle = PlayStyle.BALANCED,
        simulations: int = 5000,
    ) -> None:
        """
        Args:
            solver: EquitySolver instance (created if None).
            play_style: Default play style.
            simulations: Monte Carlo simulation count.
        """
        self.solver = solver or EquitySolver(default_simulations=simulations)
        self.play_style = play_style
        self.simulations = simulations

    def recommend(
        self,
        state: TableState,
        state_confidence: float = 1.0,
        vision_confidence: float = 1.0,
        ocr_confidence: float = 1.0,
    ) -> Recommendation:
        """
        Produce a full recommendation from current table state.
        
        Args:
            state: Canonical table state.
            state_confidence: State reconstruction confidence.
            vision_confidence: Average vision detection confidence.
            ocr_confidence: Average OCR confidence.
            
        Returns:
            Structured Recommendation with all actions scored.
        """
        hero = state.hero
        if not hero or len(hero.hole_cards) < 2:
            return self._uncertain_recommendation(
                "Карты героя не определены",
                vision_confidence,
                ocr_confidence,
                state_confidence,
                state.street,
            )

        if not all(c.is_known for c in hero.hole_cards):
            return self._uncertain_recommendation(
                "Карты героя не распознаны полностью",
                vision_confidence,
                ocr_confidence,
                state_confidence,
                state.street,
            )

        # ── Compute opponent range
        estimated_range_set = estimate_opponent_range(state, self.play_style)
        estimated_range_cards = range_to_cards(estimated_range_set)

        # ── Compute core metrics
        num_opponents = max(1, state.num_active_players - 1)

        equity = self.solver.compute_equity_vs_range(
            hero.hole_cards,
            state.community_cards,
            opponent_range_cards=estimated_range_cards,
            num_opponents=num_opponents,
            simulations=self.simulations,
        )

        hand_strength = self.solver.compute_hand_strength(
            hero.hole_cards,
            state.community_cards,
        )

        # Estimate to_call (Phase 1: use biggest opponent bet)
        max_bet = max(
            (p.bet for p in state.players if not p.is_hero and p.is_active),
            default=0.0,
        )
        to_call = max(0.0, max_bet - hero.bet)

        pot_odds = self.solver.compute_pot_odds(state.pot, to_call)
        spr = state.spr
        effective_stack_bb = state.effective_stack / state.big_blind if state.big_blind > 0 else 0.0

        # ── Check for short stack push/fold
        if effective_stack_bb <= self.SHORT_STACK_THRESHOLD_BB and state.street == Street.PREFLOP:
            return self._push_fold_recommendation(
                equity, hand_strength, pot_odds, spr, effective_stack_bb,
                state, vision_confidence, ocr_confidence, state_confidence,
            )

        # ── Score all actions
        style_adj = self.STYLE_ADJUSTMENTS[self.play_style]
        actions = self._score_actions(
            equity, pot_odds, hand_strength, spr, to_call, state, style_adj
        )

        # ── Build confidence report
        rec_confidence = self._compute_recommendation_confidence(
            equity, hand_strength, state_confidence
        )
        confidence = ConfidenceReport(
            vision_confidence=vision_confidence,
            ocr_confidence=ocr_confidence,
            state_confidence=state_confidence,
            recommendation_confidence=rec_confidence,
        )

        # ── Sort by score, best first
        actions.sort(key=lambda a: a.score, reverse=True)
        best = actions[0] if actions else Action(action_type=ActionType.UNCERTAIN)

        return Recommendation(
            best_action=best,
            all_actions=actions,
            hand_strength=hand_strength,
            equity=equity,
            pot_odds=pot_odds,
            spr=spr,
            effective_stack_bb=effective_stack_bb,
            estimated_range=list(estimated_range_set),
            confidence=confidence,
            play_style=self.play_style,
            is_uncertain=confidence.is_dangerous,
            street=state.street,
        )

    def _score_actions(
        self,
        equity: float,
        pot_odds: float,
        hand_strength: float,
        spr: float,
        to_call: float,
        state: TableState,
        style_adj: float,
    ) -> list[Action]:
        """Score all possible actions based on poker math."""
        actions: list[Action] = []

        # Fold is always available (score 0 EV)
        fold_score = 0.0
        if to_call > 0:
            # Fold has value when equity is bad relative to pot odds
            fold_score = max(0.0, pot_odds - equity + style_adj) * 0.8

        actions.append(Action(
            action_type=ActionType.FOLD,
            amount=0.0,
            score=fold_score,
            ev=-to_call if to_call > 0 else 0.0,
        ))

        # Check (free, always good if available)
        if to_call == 0:
            check_score = 0.5  # Neutral — better than fold, but not aggressive
            if hand_strength < 0.3:
                check_score = 0.7  # Weak hand, checking is good
            actions.append(Action(
                action_type=ActionType.CHECK,
                amount=0.0,
                score=check_score,
                ev=0.0,
            ))

        # Call
        if to_call > 0:
            # Call is good when equity > pot odds
            call_ev = equity * (state.pot + to_call) - to_call
            call_score = max(0.0, min(1.0, equity - pot_odds + 0.1 - style_adj))
            actions.append(Action(
                action_type=ActionType.CALL,
                amount=to_call,
                score=call_score,
                ev=call_ev,
            ))

        # Bet / Raise
        if equity > 0.55 - style_adj or hand_strength >= 0.5:
            bet_amount = self._compute_bet_size(state, equity, spr)
            bet_ev = equity * (state.pot + bet_amount) - bet_amount * (1 - equity)
            bet_score = max(0.0, min(1.0, (equity - 0.45 + style_adj * -1) * 2))

            action_type = ActionType.RAISE if to_call > 0 else ActionType.BET
            actions.append(Action(
                action_type=action_type,
                amount=bet_amount,
                score=bet_score,
                ev=bet_ev,
            ))

        # All-in
        hero = state.hero
        if hero and equity > 0.7 - style_adj and spr < 4:
            all_in_ev = equity * (state.pot + hero.stack) - hero.stack * (1 - equity)
            all_in_score = max(0.0, min(1.0, (equity - 0.6) * 3))
            actions.append(Action(
                action_type=ActionType.ALL_IN,
                amount=hero.stack if hero else 0.0,
                score=all_in_score,
                ev=all_in_ev,
            ))

        return actions

    def _compute_bet_size(
        self,
        state: TableState,
        equity: float,
        spr: float,
    ) -> float:
        """Compute optimal bet size based on pot, equity, and SPR."""
        pot = max(state.pot, state.big_blind)

        if spr < 2:
            # Low SPR: pot-sized or all-in
            return pot
        elif equity > 0.75:
            # Strong hand: 75-100% pot
            return pot * 0.85
        elif equity > 0.6:
            # Good hand: 50-75% pot
            return pot * 0.65
        else:
            # Marginal: small bet
            return pot * 0.4

    def _push_fold_recommendation(
        self,
        equity: float,
        hand_strength: float,
        pot_odds: float,
        spr: float,
        effective_stack_bb: float,
        state: TableState,
        vision_conf: float,
        ocr_conf: float,
        state_conf: float,
    ) -> Recommendation:
        """Short stack push/fold strategy."""
        # Simple push threshold based on stack depth
        push_threshold = 0.4 + (effective_stack_bb / 100)  # Tighter with deeper stack

        if equity >= push_threshold:
            best = Action(
                action_type=ActionType.ALL_IN,
                amount=state.hero.stack if state.hero else 0.0,
                score=min(1.0, equity * 1.3),
                ev=equity * state.pot - (1 - equity) * (state.hero.stack if state.hero else 0),
            )
            explanation = (
                f"Short stack ({effective_stack_bb:.0f}BB). "
                f"Equity {equity:.0%} >= порог {push_threshold:.0%}. Push."
            )
        else:
            best = Action(
                action_type=ActionType.FOLD,
                amount=0.0,
                score=1.0 - equity,
                ev=0.0,
            )
            explanation = (
                f"Short stack ({effective_stack_bb:.0f}BB). "
                f"Equity {equity:.0%} < порог {push_threshold:.0%}. Fold."
            )

        rec_conf = self._compute_recommendation_confidence(equity, hand_strength, state_conf)

        # Generate the estimated range again since we aren't passing it down,
        # or we could just use the fact that it's short-stack Preflop Push/Fold scenario.
        # It's better to log the same range. We can quickly recompute it for the recommendation.
        estimated_range_set = estimate_opponent_range(state, self.play_style)

        return Recommendation(
            best_action=best,
            all_actions=[best],
            hand_strength=hand_strength,
            equity=equity,
            pot_odds=pot_odds,
            spr=spr,
            effective_stack_bb=effective_stack_bb,
            estimated_range=list(estimated_range_set),
            confidence=ConfidenceReport(
                vision_confidence=vision_conf,
                ocr_confidence=ocr_conf,
                state_confidence=state_conf,
                recommendation_confidence=rec_conf,
            ),
            explanation=explanation,
            play_style=self.play_style,
            is_uncertain=False,
            street=state.street,
        )

    def _uncertain_recommendation(
        self,
        reason: str,
        vision_conf: float,
        ocr_conf: float,
        state_conf: float,
        street: Street,
    ) -> Recommendation:
        """Return an uncertain recommendation when state is unreliable."""
        return Recommendation(
            best_action=Action(action_type=ActionType.UNCERTAIN, score=0.0),
            hand_strength=0.0,
            equity=0.0,
            pot_odds=0.0,
            spr=0.0,
            effective_stack_bb=0.0,
            confidence=ConfidenceReport(
                vision_confidence=vision_conf,
                ocr_confidence=ocr_conf,
                state_confidence=state_conf,
                recommendation_confidence=0.0,
            ),
            explanation=f"⚠️ Uncertain: {reason}",
            play_style=self.play_style,
            is_uncertain=True,
            street=street,
        )

    @staticmethod
    def _compute_recommendation_confidence(
        equity: float,
        hand_strength: float,
        state_confidence: float,
    ) -> float:
        """
        How confident are we in the recommendation itself?
        
        High confidence when:
        - State is clear
        - Equity is extreme (clearly good or bad)
        - Decision is not marginal
        """
        # Equity clarity: extreme values = clearer decision
        equity_clarity = abs(equity - 0.5) * 2  # 0 at 50%, 1 at 0% or 100%

        return min(1.0, (state_confidence * 0.5 + equity_clarity * 0.3 + hand_strength * 0.2))
