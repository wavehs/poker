"""Shared test fixtures and configuration."""

import numpy as np
import pytest

from libs.common.schemas import (
    Card,
    PlayerState,
    Rank,
    Street,
    Suit,
    TableState,
)


@pytest.fixture
def blank_frame() -> np.ndarray:
    """A 1920x1080 blank black frame."""
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


@pytest.fixture
def sample_hero_cards() -> list[Card]:
    """Hero has Ace of hearts and King of spades."""
    return [
        Card(rank=Rank.ACE, suit=Suit.HEARTS, confidence=0.95, source="test"),
        Card(rank=Rank.KING, suit=Suit.SPADES, confidence=0.92, source="test"),
    ]


@pytest.fixture
def sample_community_cards() -> list[Card]:
    """Flop: Ah, Kd, 7c."""
    return [
        Card(rank=Rank.ACE, suit=Suit.DIAMONDS, confidence=0.90, source="test"),
        Card(rank=Rank.KING, suit=Suit.DIAMONDS, confidence=0.88, source="test"),
        Card(rank=Rank.SEVEN, suit=Suit.CLUBS, confidence=0.85, source="test"),
    ]


@pytest.fixture
def sample_table_state(sample_hero_cards, sample_community_cards) -> TableState:
    """A sample table state for testing."""
    players = [
        PlayerState(
            seat=0,
            name="Hero",
            stack=2000.0,
            stack_confidence=0.9,
            is_active=True,
            is_hero=True,
            hole_cards=sample_hero_cards,
        ),
        PlayerState(
            seat=1,
            name="Villain 1",
            stack=3000.0,
            stack_confidence=0.85,
            is_active=True,
        ),
        PlayerState(
            seat=2,
            name="Villain 2",
            stack=1500.0,
            stack_confidence=0.8,
            is_active=True,
        ),
    ]

    return TableState(
        community_cards=sample_community_cards,
        street=Street.FLOP,
        pot=500.0,
        pot_confidence=0.9,
        players=players,
        hero_seat=0,
        dealer_seat=1,
        num_active_players=3,
        is_hand_in_progress=True,
        big_blind=10.0,
        small_blind=5.0,
    )
