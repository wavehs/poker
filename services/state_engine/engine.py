"""
State Engine — Temporal fusion and canonical state reconstruction.

Fuses detections and OCR results across frames into a stable TableState.
Implements temporal smoothing, confidence aggregation, and anomaly detection.
"""

from __future__ import annotations

from collections import defaultdict

from libs.common.schemas import (
    Card,
    Detection,
    DetectionClass,
    OCRResult,
    PlayerState,
    Rank,
    Street,
    Suit,
    TableState,
    TrackedObject,
)

# ─── Card parsing helpers ────────────────────────────────────────────────────

_RANK_MAP = {r.value: r for r in Rank if r != Rank.UNKNOWN}
_SUIT_MAP = {s.value: s for s in Suit if s != Suit.UNKNOWN}


def parse_card(code: str) -> Card:
    """Parse a card code like 'Ah' into a Card object."""
    code = code.strip()
    if len(code) < 2:
        return Card()
    rank = _RANK_MAP.get(code[0].upper(), Rank.UNKNOWN)
    suit = _SUIT_MAP.get(code[1].lower(), Suit.UNKNOWN)
    return Card(rank=rank, suit=suit, source="vision")


class StateEngine:
    """
    Reconstructs canonical poker table state from vision + OCR outputs.
    
    Features:
    - Temporal smoothing over a sliding window
    - Card deduplication
    - Confidence aggregation
    - Street detection from community card count
    - Anomaly flagging
    """

    def __init__(
        self,
        smoothing_window: int = 5,
        stability_threshold: int = 3,
        hero_seat: int = 0,
    ) -> None:
        """
        Args:
            smoothing_window: Number of frames to keep in history.
            stability_threshold: Frames a detection must persist to be 'stable'.
            hero_seat: Default hero seat index.
        """
        self.smoothing_window = smoothing_window
        self.stability_threshold = stability_threshold
        self.hero_seat = hero_seat

        # Tracking state
        self._frame_history: list[dict] = []
        self._card_history: dict[str, list[str]] = defaultdict(list)  # position -> [code, ...]
        self._next_track_id: int = 0

    def update(
        self,
        detections: list[Detection],
        ocr_results: list[OCRResult],
        frame_idx: int = 0,
        timestamp_ms: float = 0.0,
        tracked_objects: list[TrackedObject] | None = None,
        frame_shape: tuple[int, int] | None = None,
    ) -> tuple[TableState, list[TrackedObject]]:
        """
        Process new frame data and produce updated canonical state.

        Args:
            detections: Vision detections for this frame.
            ocr_results: OCR results for this frame.
            frame_idx: Frame index.
            timestamp_ms: Frame timestamp.
            tracked_objects: Pre-computed tracked objects from ObjectTracker.
                             If None, simple internal tracking is used.
            frame_shape: Optional (height, width) of the source frame.
                         When provided, hero/board classification uses real
                         pixel coordinates instead of an estimated frame height.

        Returns:
            Tuple of (TableState, list of TrackedObjects).
        """
        # Store frame data
        frame_data = {
            "frame_idx": frame_idx,
            "timestamp_ms": timestamp_ms,
            "detections": detections,
            "ocr_results": ocr_results,
        }
        self._frame_history.append(frame_data)

        # Trim history
        if len(self._frame_history) > self.smoothing_window:
            self._frame_history = self._frame_history[-self.smoothing_window:]

        # Build state
        cards = self._extract_cards(detections)
        hero_cards, community_cards = self._classify_cards(
            cards, detections, frame_shape=frame_shape
        )
        pot = self._extract_pot(detections, ocr_results)
        players = self._extract_players(detections, ocr_results, frame_shape=frame_shape)
        dealer_seat = self._extract_dealer(detections, players)

        # Mark the dealer flag on the corresponding player so downstream
        # consumers don't have to cross-reference `state.dealer_seat`.
        if dealer_seat is not None:
            for p in players:
                if p.seat == dealer_seat:
                    p.is_dealer = True
                    break

        # Use external tracker if provided, else build simple tracked objects
        if tracked_objects is not None:
            tracked = tracked_objects
        else:
            tracked = self._build_tracked_objects(detections)

        # Determine street
        street = self._determine_street(len(community_cards))

        # Build state
        state = TableState(
            frame_idx=frame_idx,
            timestamp_ms=timestamp_ms,
            community_cards=community_cards,
            street=street,
            pot=pot,
            pot_confidence=self._compute_pot_confidence(ocr_results),
            players=players,
            hero_seat=next((p.seat for p in players if p.is_hero), -1),
            dealer_seat=dealer_seat,
            num_active_players=len([p for p in players if p.is_active]),
            is_hand_in_progress=len(hero_cards) == 2 or len(community_cards) > 0,
        )

        # Set hero cards
        if state.hero is not None:
            state.hero.hole_cards = hero_cards

        return state, tracked

    def _extract_cards(self, detections: list[Detection]) -> list[tuple[Card, Detection]]:
        """Extract and parse all card detections."""
        cards: list[tuple[Card, Detection]] = []
        seen_codes: set[str] = set()

        for det in detections:
            if det.detection_class != DetectionClass.CARD:
                continue
            card = parse_card(det.label)
            card.confidence = det.bbox.confidence
            if card.is_known and card.code not in seen_codes:
                seen_codes.add(card.code)
                cards.append((card, det))

        return cards

    HERO_CARD_Y_THRESHOLD = 0.65

    def _frame_height(
        self,
        detections: list[Detection],
        frame_shape: tuple[int, int] | None,
    ) -> float:
        """Return frame height in pixels, preferring explicit `frame_shape`."""
        if frame_shape is not None and frame_shape[0] > 0:
            return float(frame_shape[0])
        # Fallback: estimate from detections (legacy behavior).
        frame_h = 1080.0
        for det in detections:
            bottom = det.bbox.y + det.bbox.h
            if bottom > frame_h:
                frame_h = bottom + 100.0
        return frame_h

    def _classify_cards(
        self,
        cards: list[tuple[Card, Detection]],
        detections: list[Detection],
        frame_shape: tuple[int, int] | None = None,
    ) -> tuple[list[Card], list[Card]]:
        """
        Classify cards as hero hole cards vs community cards based on Y-position.

        Hero cards live in the bottom band of the frame; community cards live
        in the middle band. Uses the real frame height when available, falling
        back to a detection-derived estimate.
        """
        if not cards:
            return [], []

        frame_h = self._frame_height(detections, frame_shape)

        hero_cards: list[Card] = []
        community_cards: list[Card] = []

        for card, det in cards:
            relative_y = (det.bbox.y + det.bbox.h / 2) / frame_h
            if relative_y > self.HERO_CARD_Y_THRESHOLD:
                card.source = "vision_hero"
                hero_cards.append(card)
            else:
                card.source = "vision_board"
                community_cards.append(card)

        # Limit to valid counts
        hero_cards = sorted(hero_cards, key=lambda c: c.confidence, reverse=True)[:2]
        community_cards = sorted(community_cards, key=lambda c: c.confidence, reverse=True)[:5]

        return hero_cards, community_cards

    def _extract_pot(
        self,
        detections: list[Detection],
        ocr_results: list[OCRResult],
    ) -> float:
        """Extract pot size from OCR results or detection labels."""
        # Try OCR first
        for ocr in ocr_results:
            if ocr.field_type == "pot":
                try:
                    return float(ocr.text)
                except ValueError:
                    continue

        # Fallback to detection labels
        for det in detections:
            if det.detection_class == DetectionClass.POT:
                try:
                    return float(det.label)
                except ValueError:
                    continue

        return 0.0

    @staticmethod
    def _panel_center(det: Detection) -> tuple[float, float]:
        return det.bbox.x + det.bbox.w / 2.0, det.bbox.y + det.bbox.h / 2.0

    def _extract_players(
        self,
        detections: list[Detection],
        ocr_results: list[OCRResult],
        frame_shape: tuple[int, int] | None = None,
    ) -> list[PlayerState]:
        """Build player states from detected panels and OCR.

        Hero is identified as the panel whose vertical center is closest to
        the bottom edge of the frame (most poker clients render the hero at
        the bottom of the table). If only one panel is detected, it is the
        hero. Falls back to ``self.hero_seat`` index when no spatial info is
        available.
        """
        players: list[PlayerState] = []
        panel_detections = [
            d for d in detections
            if d.detection_class == DetectionClass.PLAYER_PANEL
        ]

        stack_ocr_results = [
            ocr for ocr in ocr_results
            if ocr.field_type == "stack"
        ]

        frame_h = self._frame_height(detections, frame_shape)
        # Identify hero by maximum Y-center (closest to bottom).
        hero_idx = -1
        if panel_detections:
            hero_idx = max(
                range(len(panel_detections)),
                key=lambda i: self._panel_center(panel_detections[i])[1],
            )
            # Require hero panel to be in the bottom band; otherwise fall back.
            hero_y = self._panel_center(panel_detections[hero_idx])[1] / frame_h
            if hero_y < 0.55:
                hero_idx = -1

        for i, panel in enumerate(panel_detections):
            stack = 0.0
            stack_conf = 0.0

            try:
                stack = float(panel.label)
                stack_conf = panel.bbox.confidence
            except (ValueError, TypeError):
                pass

            for ocr in stack_ocr_results:
                try:
                    stack = float(ocr.text)
                    stack_conf = ocr.confidence
                    break
                except ValueError:
                    continue

            is_hero = (i == hero_idx) if hero_idx >= 0 else (i == self.hero_seat)
            player = PlayerState(
                seat=i,
                name=f"Player {i + 1}",
                stack=stack,
                stack_confidence=stack_conf,
                is_active=True,
                is_hero=is_hero,
                is_dealer=False,
            )
            players.append(player)

        if not any(p.is_hero for p in players) and players:
            players[0].is_hero = True

        return players

    def _extract_dealer(
        self,
        detections: list[Detection],
        players: list[PlayerState] | None = None,
    ) -> int:
        """Find dealer button position and map it to the nearest player seat.

        Uses Euclidean distance between the button center and each player
        panel center. Returns the seat index of the closest player, or -1 if
        no button was detected or no panels are available.
        """
        button_det = next(
            (d for d in detections if d.detection_class == DetectionClass.DEALER_BUTTON),
            None,
        )
        if button_det is None or not players:
            return -1

        panel_dets = [d for d in detections if d.detection_class == DetectionClass.PLAYER_PANEL]
        if not panel_dets or len(panel_dets) != len(players):
            return -1

        bx, by = self._panel_center(button_det)

        best_seat = -1
        best_dist_sq = float("inf")
        for player, panel in zip(players, panel_dets, strict=False):
            px, py = self._panel_center(panel)
            dx = px - bx
            dy = py - by
            dist_sq = dx * dx + dy * dy
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_seat = player.seat
        return best_seat

    _STREET_BY_COMMUNITY_CARDS: dict[int, Street] = {
        0: Street.PREFLOP,
        3: Street.FLOP,
        4: Street.TURN,
        5: Street.RIVER,
    }

    def _determine_street(self, num_community: int) -> Street:
        """Determine current street from number of community cards."""
        return self._STREET_BY_COMMUNITY_CARDS.get(num_community, Street.UNKNOWN)

    def _compute_pot_confidence(self, ocr_results: list[OCRResult]) -> float:
        """Compute confidence for pot extraction."""
        pot_ocrs = [o for o in ocr_results if o.field_type == "pot"]
        if not pot_ocrs:
            return 0.5  # No OCR, using detection label
        return max(o.confidence for o in pot_ocrs)

    def _build_tracked_objects(self, detections: list[Detection]) -> list[TrackedObject]:
        """Build tracked objects from current frame (simplified Phase 1)."""
        tracked: list[TrackedObject] = []
        for det in detections:
            obj = TrackedObject(
                track_id=self._next_track_id,
                detection_class=det.detection_class,
                detections=[det],
                smoothed_label=det.label,
                smoothed_confidence=det.bbox.confidence,
                is_stable=True,  # Phase 1: assume stable
                frames_seen=1,
            )
            tracked.append(obj)
            self._next_track_id += 1
        return tracked

    def get_state_confidence(self, state: TableState) -> float:
        """
        Compute overall state reconstruction confidence.
        
        Factors:
        - Are hero cards detected?
        - Is pot detected?
        - Are players detected?
        - Is street consistent?
        """
        scores: list[float] = []

        # Hero cards
        hero = state.hero
        if hero and len(hero.hole_cards) == 2:
            avg_card_conf = sum(c.confidence for c in hero.hole_cards) / 2
            scores.append(avg_card_conf)
        else:
            scores.append(0.3)

        # Pot
        if state.pot > 0:
            scores.append(state.pot_confidence)
        else:
            scores.append(0.4)

        # Players
        if state.num_active_players >= 2:
            scores.append(0.9)
        else:
            scores.append(0.3)

        # Street consistency
        if state.street != Street.UNKNOWN:
            scores.append(0.9)
        else:
            scores.append(0.5)

        return sum(scores) / len(scores) if scores else 0.0
