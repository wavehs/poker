# ♠️ Poker Helper — Real-Time External Poker Assistant

> Production-grade external poker assistant using screen capture, computer vision, OCR, and deterministic poker logic.

## ⚠️ Ethical Use

This tool is **fully external** — it does NOT:
- Read process memory
- Inject into game processes
- Hook internal APIs
- Automate clicks or gameplay actions
- Implement stealth or anti-detection

It ONLY uses screen capture, computer vision, OCR, and deterministic poker math, displayed in a separate helper UI window.

## Architecture

```
Screen Capture → Vision (YOLO) → OCR → State Engine → Solver → Policy → UI
```

Each module communicates via typed Pydantic schemas. All confidence scores are first-class citizens.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run API server
uvicorn apps.api.main:app --reload --port 8000

# Run UI (dev mode)
python -m http.server 3000 --directory apps/ui

# Run tests
pytest tests/ -v
```

## Docker

```bash
docker-compose up --build
```

## Project Structure

| Directory | Purpose |
|-----------|---------|
| `libs/common/` | Shared Pydantic schemas |
| `services/capture_agent/` | Screen/window capture |
| `services/vision_core/` | YOLO card/element detection |
| `services/ocr_core/` | OCR text extraction |
| `services/state_engine/` | Temporal fusion → canonical state |
| `services/solver_core/` | Equity, Monte Carlo, pot odds |
| `services/policy_layer/` | Action recommendation engine |
| `services/explainer/` | Human-readable explanations |
| `apps/api/` | FastAPI backend |
| `apps/ui/` | Helper UI overlay |
| `data/` | Frames, labels, synthetic data |
| `tests/` | Unit & integration tests |
| `evals/` | Evaluation framework |
| `infra/` | Docker, nginx configs |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/analyze-frame` | POST | Analyze single frame |
| `/api/v1/analyze-sequence` | POST | Analyze frame sequence |

## License

Private — for personal use only.
