"""
Synthetic Poker Table Frame Generator.

Generates simple synthetic poker table images for development and testing
before real training data is available.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

# ─── Constants ────────────────────────────────────────────────────────────────

RANKS = "23456789TJQKA"
SUITS = "hdcs"
SUIT_SYMBOLS = {"h": "♥", "d": "♦", "c": "♣", "s": "♠"}
SUIT_COLORS = {
    "h": (0, 0, 220),    # Red (BGR)
    "d": (0, 0, 220),    # Red
    "c": (40, 40, 40),   # Black
    "s": (40, 40, 40),   # Black
}

TABLE_COLOR = (30, 100, 50)          # Dark green felt (BGR)
TABLE_OUTLINE = (40, 130, 70)        # Lighter green outline
CARD_BG = (245, 245, 245)            # White card background
TEXT_COLOR = (20, 20, 20)            # Dark text
POT_COLOR = (0, 200, 255)           # Gold/amber for pot text
PANEL_COLOR = (50, 50, 60)          # Dark gray panels
PANEL_TEXT = (200, 200, 210)        # Light panel text


def generate_synthetic_frame(
    width: int = 1920,
    height: int = 1080,
    num_players: int = 6,
    num_community: int | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Generate a synthetic poker table frame.
    
    Args:
        width: Frame width.
        height: Frame height.
        num_players: Number of players (2-9).
        num_community: Number of community cards (None = random).
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (frame_bgr, metadata_dict).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    frame = np.full((height, width, 3), TABLE_COLOR, dtype=np.uint8)
    metadata: dict = {"cards": [], "players": [], "pot": 0, "community": []}

    # ── Draw table ellipse
    cx, cy = width // 2, height // 2
    rx, ry = int(width * 0.38), int(height * 0.35)
    cv2.ellipse(frame, (cx, cy), (rx, ry), 0, 0, 360, TABLE_OUTLINE, 3)
    cv2.ellipse(frame, (cx, cy), (rx - 5, ry - 5), 0, 0, 360, TABLE_COLOR, -1)

    # ── Deck management
    deck = [(r, s) for r in RANKS for s in SUITS]
    random.shuffle(deck)
    deck_idx = 0

    # ── Community cards
    if num_community is None:
        num_community = random.choice([0, 3, 4, 5])

    community_cards = []
    card_w, card_h = 55, 80
    x_start = cx - (num_community * (card_w + 8)) // 2

    for i in range(num_community):
        if deck_idx >= len(deck):
            break
        rank, suit = deck[deck_idx]
        deck_idx += 1
        card_code = f"{rank}{suit}"
        community_cards.append(card_code)
        x = x_start + i * (card_w + 8)
        y = cy - card_h // 2 - 20
        _draw_card(frame, x, y, card_w, card_h, rank, suit)

    metadata["community"] = community_cards

    # ── Pot
    pot = random.choice([0, 50, 100, 250, 500, 1000, 2500, 5000])
    metadata["pot"] = pot
    if pot > 0:
        pot_text = f"Pot: {pot}"
        text_size = cv2.getTextSize(pot_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        tx = cx - text_size[0] // 2
        ty = cy + card_h // 2 + 30
        cv2.putText(frame, pot_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, POT_COLOR, 2)

    # ── Players
    num_players = min(num_players, 9)
    import math
    for i in range(num_players):
        angle = (2 * math.pi * i / num_players) - math.pi / 2
        px = int(cx + rx * 1.15 * math.cos(angle))
        py = int(cy + ry * 1.15 * math.sin(angle))

        stack = random.choice([500, 1000, 1500, 2000, 3000, 5000, 10000])
        name = f"Player {i + 1}"
        is_hero = (i == 0)

        # Draw player panel
        panel_w, panel_h = 140, 50
        panel_x = max(5, min(px - panel_w // 2, width - panel_w - 5))
        panel_y = max(5, min(py - panel_h // 2, height - panel_h - 5))

        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            PANEL_COLOR, -1
        )
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (80, 80, 90), 1
        )

        # Name
        cv2.putText(
            frame, name,
            (panel_x + 8, panel_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, PANEL_TEXT, 1
        )
        # Stack
        cv2.putText(
            frame, str(stack),
            (panel_x + 8, panel_y + 38),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 220, 100), 1
        )

        player_info = {
            "seat": i,
            "name": name,
            "stack": stack,
            "is_hero": is_hero,
            "position": (panel_x, panel_y),
        }

        # Hero cards
        if is_hero and deck_idx + 1 < len(deck):
            hero_cards = []
            for j in range(2):
                rank, suit = deck[deck_idx]
                deck_idx += 1
                card_code = f"{rank}{suit}"
                hero_cards.append(card_code)
                hx = cx - 50 + j * (card_w + 8)
                hy = height - card_h - 80
                _draw_card(frame, hx, hy, card_w, card_h, rank, suit)
            player_info["hole_cards"] = hero_cards

        metadata["players"].append(player_info)

    # ── Dealer button
    dealer_idx = random.randint(0, num_players - 1)
    metadata["dealer_seat"] = dealer_idx

    return frame, metadata


def _draw_card(
    frame: np.ndarray,
    x: int, y: int,
    w: int, h: int,
    rank: str, suit: str,
) -> None:
    """Draw a playing card on the frame."""
    # Card background
    cv2.rectangle(frame, (x, y), (x + w, y + h), CARD_BG, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (180, 180, 180), 1)

    # Rounded corners (simplified)
    color = SUIT_COLORS.get(suit, (0, 0, 0))
    symbol = SUIT_SYMBOLS.get(suit, "?")

    # Rank text
    cv2.putText(
        frame, rank,
        (x + 5, y + 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
    )

    # Suit symbol (using simple text)
    cv2.putText(
        frame, suit.upper(),
        (x + 5, y + 42),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
    )

    # Center rank
    cv2.putText(
        frame, rank,
        (x + w // 2 - 8, y + h // 2 + 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
    )


def generate_dataset(
    output_dir: str = "data/synthetic_tables/output",
    num_frames: int = 20,
    width: int = 1920,
    height: int = 1080,
) -> None:
    """
    Generate a dataset of synthetic poker table frames.
    
    Args:
        output_dir: Directory to save frames.
        num_frames: Number of frames to generate.
        width: Frame width.
        height: Frame height.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_frames} synthetic frames in {out}...")

    for i in range(num_frames):
        frame, meta = generate_synthetic_frame(
            width=width,
            height=height,
            seed=i * 42,
        )

        filename = f"frame_{i:04d}.png"
        cv2.imwrite(str(out / filename), frame)

        # Save metadata
        import json
        meta_file = out / f"frame_{i:04d}.json"
        meta_file.write_text(json.dumps(meta, indent=2))

    print(f"✅ Generated {num_frames} frames in {out}")


if __name__ == "__main__":
    generate_dataset()
