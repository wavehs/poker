"""
Poker Helper — Canonical Shared Schemas.

All modules communicate via these typed Pydantic models.
Confidence scoring is a first-class citizen at every layer.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

# ─── Enumerations ────────────────────────────────────────────────────────────


class Suit(str, Enum):
    HEARTS = "h"
    DIAMONDS = "d"
    CLUBS = "c"
    SPADES = "s"
    UNKNOWN = "?"


class Rank(str, Enum):
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "T"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    ACE = "A"
    UNKNOWN = "?"


class DetectionClass(str, Enum):
    """Classes that the vision model can detect."""
    CARD = "card"
    CHIP_STACK = "chip_stack"
    POT = "pot"
    DEALER_BUTTON = "dealer_button"
    PLAYER_PANEL = "player_panel"
    BET_AMOUNT = "bet_amount"
    ACTION_BUTTON = "action_button"
    BOARD_AREA = "board_area"


class Street(str, Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"
    UNKNOWN = "unknown"


class ActionType(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"
    UNCERTAIN = "uncertain"


class PlayStyle(str, Enum):
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"


# ─── Vision / Detection Layer ────────────────────────────────────────────────


class BoundingBox(BaseModel):
    """Pixel-space bounding box from detection model."""
    x: float = Field(..., description="Top-left X coordinate (px)")
    y: float = Field(..., description="Top-left Y coordinate (px)")
    w: float = Field(..., description="Width (px)")
    h: float = Field(..., description="Height (px)")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detection confidence [0,1]"
    )


class Detection(BaseModel):
    """Single object detected in a frame."""
    detection_class: DetectionClass
    bbox: BoundingBox
    label: str = Field(default="", description="Predicted label (e.g. 'Ah', '500')")
    frame_idx: int = Field(default=0, description="Source frame index")
    timestamp_ms: float = Field(default=0.0, description="Frame timestamp in ms")


class OCRResult(BaseModel):
    """Result from OCR engine for a text region."""
    text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox
    field_type: str = Field(
        default="generic",
        description="Semantic type: 'pot', 'stack', 'bet', 'blind', 'player_name'",
    )


class TrackedObject(BaseModel):
    """Object tracked across multiple frames with temporal smoothing."""
    track_id: int
    detection_class: DetectionClass
    detections: list[Detection] = Field(default_factory=list)
    smoothed_label: str = Field(
        default="", description="Label after temporal consensus"
    )
    smoothed_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Smoothed confidence"
    )
    is_stable: bool = Field(
        default=False, description="True if label is stable across N frames"
    )
    frames_seen: int = Field(default=0, description="Number of frames observed")


# ─── Poker Domain Layer ─────────────────────────────────────────────────────


class Card(BaseModel):
    """A single playing card."""
    rank: Rank = Rank.UNKNOWN
    suit: Suit = Suit.UNKNOWN
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: str = Field(
        default="unknown", description="Detection source: 'vision', 'ocr', 'fused'"
    )

    @property
    def code(self) -> str:
        """Short code like 'Ah', 'Tc', '??'."""
        return f"{self.rank.value}{self.suit.value}"

    @property
    def is_known(self) -> bool:
        return self.rank != Rank.UNKNOWN and self.suit != Suit.UNKNOWN

    def __str__(self) -> str:
        return self.code


class OpponentProfile(BaseModel):
    """Statistical profile of an opponent."""
    seat_id: int
    hands_seen: int = Field(default=0, ge=0)
    vpip: float = Field(default=0.0, ge=0.0, le=1.0)
    pfr: float = Field(default=0.0, ge=0.0, le=1.0)
    af: float = Field(default=0.0, ge=0.0)
    fold_to_cbet: float = Field(default=0.0, ge=0.0, le=1.0)
    sizing_tells: dict[Street, float] = Field(default_factory=dict)
    sample_size: int = Field(default=0, ge=0)


class PlayerState(BaseModel):
    """State of a single player at the table."""
    seat: int = Field(..., ge=0, le=9, description="Seat index 0-9")
    name: str = Field(default="Unknown")
    stack: float = Field(default=0.0, ge=0.0, description="Chip stack")
    stack_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    bet: float = Field(default=0.0, ge=0.0, description="Current bet in this round")
    is_active: bool = Field(default=True, description="Still in the hand")
    is_dealer: bool = Field(default=False)
    is_hero: bool = Field(default=False, description="True if this is the user's seat")
    hole_cards: list[Card] = Field(default_factory=list, max_length=2)
    has_acted: bool = Field(default=False)
    last_action: ActionType | None = None


class TableState(BaseModel):
    """Canonical poker table state reconstructed from visual input."""
    # ── Identity
    table_id: str = Field(default="default")
    frame_idx: int = Field(default=0)
    timestamp_ms: float = Field(default=0.0)

    # ── Board & Cards
    community_cards: list[Card] = Field(default_factory=list, max_length=5)
    street: Street = Street.UNKNOWN

    # ── Money
    pot: float = Field(default=0.0, ge=0.0)
    pot_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    small_blind: float = Field(default=0.5)
    big_blind: float = Field(default=1.0)

    # ── Players
    players: list[PlayerState] = Field(default_factory=list)
    hero_seat: int = Field(default=-1, description="Seat index of the hero (-1 = unknown)")
    dealer_seat: int = Field(default=-1)

    # ── Meta
    num_active_players: int = Field(default=0)
    is_hand_in_progress: bool = Field(default=False)

    @property
    def hero(self) -> PlayerState | None:
        """Get the hero player, if identified."""
        for p in self.players:
            if p.is_hero:
                return p
        return None

    @property
    def effective_stack(self) -> float:
        """Effective stack: min of hero stack and max opponent stack."""
        hero = self.hero
        if not hero:
            return 0.0
        opponents = [p.stack for p in self.players if p.is_active and not p.is_hero]
        if not opponents:
            return hero.stack
        return min(hero.stack, max(opponents))

    @property
    def spr(self) -> float:
        """Stack-to-pot ratio."""
        if self.pot <= 0:
            return float("inf")
        return self.effective_stack / self.pot


# ─── Recommendation Layer ────────────────────────────────────────────────────


class Action(BaseModel):
    """A possible action with computed score."""
    action_type: ActionType
    amount: float = Field(default=0.0, ge=0.0, description="Bet/raise amount if applicable")
    score: float = Field(
        default=0.0, description="Policy score [0,1] — higher is better"
    )
    ev: float = Field(default=0.0, description="Expected value estimate")


class ConfidenceReport(BaseModel):
    """Multi-layer confidence breakdown."""
    vision_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    ocr_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    state_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    recommendation_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def overall(self) -> float:
        """Geometric mean of all confidence scores."""
        scores = [
            self.vision_confidence,
            self.ocr_confidence,
            self.state_confidence,
            self.recommendation_confidence,
        ]
        product = 1.0
        for s in scores:
            product *= max(s, 1e-9)
        return product ** (1.0 / len(scores))

    @property
    def is_dangerous(self) -> bool:
        """True when overall confidence is too low to trust."""
        return self.overall < 0.5 or any(
            s < 0.3
            for s in [
                self.vision_confidence,
                self.ocr_confidence,
                self.state_confidence,
            ]
        )


class Recommendation(BaseModel):
    """Final structured recommendation for the player."""
    best_action: Action
    all_actions: list[Action] = Field(default_factory=list)
    hand_strength: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Relative hand strength [0,1]"
    )
    equity: float = Field(default=0.0, ge=0.0, le=1.0, description="Monte Carlo equity [0,1]")
    pot_odds: float = Field(default=0.0, ge=0.0, description="Pot odds ratio")
    spr: float = Field(default=0.0, ge=0.0, description="Stack-to-pot ratio")
    effective_stack_bb: float = Field(default=0.0, ge=0.0, description="Effective stack in BBs")
    confidence: ConfidenceReport = Field(default_factory=ConfidenceReport)
    explanation: str = Field(default="", description="Human-readable explanation")
    play_style: PlayStyle = Field(default=PlayStyle.BALANCED)
    is_uncertain: bool = Field(
        default=False, description="True when system can't produce reliable advice"
    )
    street: Street = Street.UNKNOWN


class FrameAnalysis(BaseModel):
    """Complete analysis result for a single frame."""
    frame_idx: int = Field(default=0)
    timestamp_ms: float = Field(default=0.0)
    detections: list[Detection] = Field(default_factory=list)
    ocr_results: list[OCRResult] = Field(default_factory=list)
    tracked_objects: list[TrackedObject] = Field(default_factory=list)
    table_state: TableState = Field(default_factory=TableState)
    recommendation: Recommendation = Field(default_factory=lambda: Recommendation(
        best_action=Action(action_type=ActionType.UNCERTAIN)
    ))
    processing_time_ms: float = Field(default=0.0, description="Total pipeline latency")
