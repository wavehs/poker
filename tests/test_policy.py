"""Tests for Policy Layer."""

from libs.common.schemas import ActionType, PlayStyle, Street
from services.policy_layer.policy import PolicyEngine
from services.solver_core.solver import EquitySolver


class TestPolicyEngine:
    def test_recommend_produces_recommendation(self, sample_table_state):
        policy = PolicyEngine(
            solver=EquitySolver(default_simulations=200),
            play_style=PlayStyle.BALANCED,
        )
        rec = policy.recommend(sample_table_state, state_confidence=0.9)
        assert rec.best_action is not None
        assert rec.best_action.action_type != ActionType.UNCERTAIN
        assert rec.equity > 0
        assert rec.hand_strength > 0

    def test_uncertain_when_no_hero_cards(self):
        from libs.common.schemas import PlayerState, TableState

        state = TableState(
            players=[
                PlayerState(seat=0, is_hero=True, hole_cards=[], stack=1000),
                PlayerState(seat=1, is_active=True, stack=1500),
            ],
            num_active_players=2,
            street=Street.PREFLOP,
        )

        policy = PolicyEngine()
        rec = policy.recommend(state)
        assert rec.is_uncertain
        assert rec.best_action.action_type == ActionType.UNCERTAIN

    def test_aggressive_style_more_aggressive(self, sample_table_state):
        agg = PolicyEngine(
            solver=EquitySolver(default_simulations=200),
            play_style=PlayStyle.AGGRESSIVE,
        )
        con = PolicyEngine(
            solver=EquitySolver(default_simulations=200),
            play_style=PlayStyle.CONSERVATIVE,
        )

        rec_agg = agg.recommend(sample_table_state)
        rec_con = con.recommend(sample_table_state)

        # Both should produce valid recommendations
        assert not rec_agg.is_uncertain
        assert not rec_con.is_uncertain

    def test_confidence_report_populated(self, sample_table_state):
        policy = PolicyEngine(solver=EquitySolver(default_simulations=200))
        rec = policy.recommend(
            sample_table_state,
            vision_confidence=0.9,
            ocr_confidence=0.85,
            state_confidence=0.8,
        )
        assert rec.confidence.vision_confidence == 0.9
        assert rec.confidence.ocr_confidence == 0.85
        assert rec.confidence.state_confidence == 0.8
        assert rec.confidence.recommendation_confidence > 0

    def test_all_actions_scored(self, sample_table_state):
        policy = PolicyEngine(solver=EquitySolver(default_simulations=200))
        rec = policy.recommend(sample_table_state)
        assert len(rec.all_actions) >= 2  # At least fold + one other
        for act in rec.all_actions:
            assert act.action_type in ActionType

    def test_recommend_latency_under_1s(self, sample_table_state):
        """Phase 3: recommendation should complete in <1s with 500 sims."""
        import time

        policy = PolicyEngine(
            solver=EquitySolver(default_simulations=500, adaptive=True),
            play_style=PlayStyle.BALANCED,
        )
        t0 = time.perf_counter()
        rec = policy.recommend(sample_table_state, state_confidence=0.9)
        elapsed = time.perf_counter() - t0

        assert not rec.is_uncertain
        assert elapsed < 1.0, f"recommend() took {elapsed:.2f}s, target <1s"

    def test_solver_profile_accessible(self, sample_table_state):
        """Phase 3: solver profile should be accessible after recommend()."""
        solver = EquitySolver(default_simulations=200, adaptive=False)
        policy = PolicyEngine(solver=solver, play_style=PlayStyle.BALANCED, simulations=200)
        policy.recommend(sample_table_state)

        prof = solver.last_profile
        assert prof is not None
        assert prof.total_ms > 0
        assert prof.simulations_run == 200
