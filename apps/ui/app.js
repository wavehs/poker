/**
 * Poker Helper — UI Application
 * 
 * Connects to the FastAPI backend and displays real-time
 * poker recommendations in a compact overlay.
 */

const API_BASE = 'http://localhost:8000';

// ─── State ─────────────────────────────────────────────────────────────────

const state = {
    isConnected: false,
    isAutoMode: false,
    autoInterval: null,
    lastAnalysis: null,
};

// ─── Action names (Russian) ────────────────────────────────────────────────

const ACTION_NAMES = {
    fold: 'Фолд',
    check: 'Чек',
    call: 'Колл',
    bet: 'Бет',
    raise: 'Рейз',
    all_in: 'Олл-ин',
    uncertain: 'Неизвестно',
};

const STREET_NAMES = {
    preflop: 'Префлоп',
    flop: 'Флоп',
    turn: 'Тёрн',
    river: 'Ривер',
    showdown: 'Шоудаун',
    unknown: '—',
};

const SUIT_SYMBOLS = {
    h: '♥', d: '♦', c: '♣', s: '♠',
};

const SUIT_CLASSES = {
    h: 'hearts', d: 'diamonds', c: 'clubs', s: 'spades',
};

// ─── DOM Elements ──────────────────────────────────────────────────────────

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const dom = {
    btnAnalyze: $('#btn-analyze'),
    btnAuto: $('#btn-auto'),
    connectionStatus: $('#connection-status'),
    streetBadge: $('#street-badge'),
    bestAction: $('#best-action'),
    equity: $('#equity'),
    handStrength: $('#hand-strength'),
    potOdds: $('#pot-odds'),
    spr: $('#spr'),
    effStack: $('#eff-stack'),
    potSize: $('#pot-size'),
    allActions: $('#all-actions'),
    explanation: $('#explanation'),
    heroCards: $('#hero-cards'),
    boardCards: $('#board-cards'),
    detectionCount: $('#detection-count'),
    debugLog: $('#debug-log'),
    latencyValue: $('#latency-value'),
    dangerBanner: $('#danger-banner'),
    confVision: $('#conf-vision'),
    confVisionVal: $('#conf-vision-val'),
    confOcr: $('#conf-ocr'),
    confOcrVal: $('#conf-ocr-val'),
    confState: $('#conf-state'),
    confStateVal: $('#conf-state-val'),
    confRec: $('#conf-rec'),
    confRecVal: $('#conf-rec-val'),
    potOddsIndicator: $('#pot-odds-indicator'),
    potOddsStatus: $('#pot-odds-status'),
    potOddsFill: $('#pot-odds-fill'),
    potOddsMarker: $('#pot-odds-marker'),
};

// ─── API ───────────────────────────────────────────────────────────────────

async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/health`);
        if (res.ok) {
            setConnected(true);
            return true;
        }
    } catch (e) {
        // ignore
    }
    setConnected(false);
    return false;
}

async function analyzeSynthetic() {
    try {
        dom.btnAnalyze.disabled = true;
        dom.btnAnalyze.textContent = '⏳ Анализ...';

        const res = await fetch(`${API_BASE}/api/v1/analyze-synthetic`, {
            method: 'POST',
        });

        if (!res.ok) {
            throw new Error(`HTTP ${res.status}`);
        }

        const data = await res.json();
        state.lastAnalysis = data;
        updateUI(data);
        setConnected(true);
    } catch (e) {
        console.error('Analysis failed:', e);
        setConnected(false);
        showError(`Ошибка: ${e.message}`);
    } finally {
        dom.btnAnalyze.disabled = false;
        dom.btnAnalyze.textContent = '⚡ Анализ';
    }
}

// ─── UI Updates ────────────────────────────────────────────────────────────

function setConnected(connected) {
    state.isConnected = connected;
    const dot = dom.connectionStatus.querySelector('.status-dot');
    const text = dom.connectionStatus.querySelector('span:last-child');

    if (connected) {
        dot.className = 'status-dot status-dot--connected';
        text.textContent = 'Online';
    } else {
        dot.className = 'status-dot status-dot--disconnected';
        text.textContent = 'Offline';
    }
}

function updateUI(analysis) {
    const rec = analysis.recommendation;
    const ts = analysis.table_state;
    const conf = rec.confidence;

    // ── Street badge
    dom.streetBadge.textContent = STREET_NAMES[rec.street] || '—';

    // ── Best action
    updateBestAction(rec);

    // ── Metrics
    dom.equity.textContent = formatPercent(rec.equity);
    dom.handStrength.textContent = formatPercent(rec.hand_strength);
    dom.potOdds.textContent = formatPercent(rec.pot_odds);
    dom.spr.textContent = rec.spr.toFixed(1);
    dom.effStack.textContent = `${rec.effective_stack_bb.toFixed(0)}BB`;
    dom.potSize.textContent = ts.pot.toFixed(0);

    // ── Pot Odds Indicator
    updatePotOddsIndicator(analysis);

    // ── All actions
    updateActionsList(rec.all_actions);

    // ── Explanation
    dom.explanation.querySelector('.explanation__text').textContent =
        rec.explanation || 'Нет данных';

    // ── Confidence
    updateConfidence(conf);

    // ── Cards
    updateCards(ts);

    // ── Detections
    updateDetections(analysis.detections);

    // ── Latency
    dom.latencyValue.textContent = `${analysis.processing_time_ms.toFixed(0)}ms`;

    // ── Danger banner
    if (rec.is_uncertain || conf.vision_confidence < 0.3 ||
        conf.ocr_confidence < 0.3 || conf.state_confidence < 0.3) {
        dom.dangerBanner.hidden = false;
    } else {
        dom.dangerBanner.hidden = true;
    }
}

function updateBestAction(rec) {
    const action = rec.best_action;
    const name = ACTION_NAMES[action.action_type] || action.action_type;
    const actionEl = dom.bestAction;

    // Remove old classes
    actionEl.className = 'recommendation__action';
    actionEl.classList.add(`recommendation__action--${action.action_type}`);

    let html = `<span class="recommendation__action-label">${name}</span>`;
    if (action.amount > 0) {
        html += `<span class="recommendation__action-amount">${action.amount.toFixed(0)}</span>`;
    }
    html += `<span class="recommendation__action-score">Score: ${action.score.toFixed(2)} | EV: ${action.ev >= 0 ? '+' : ''}${action.ev.toFixed(1)}</span>`;

    actionEl.innerHTML = html;
}

function updatePotOddsIndicator(analysis) {
    const rec = analysis.recommendation;

    // As per requirement, look in solver_result first if it exists, fallback to recommendation
    const potOdds = analysis.solver_result?.pot_odds ?? rec.pot_odds;
    const equity = analysis.solver_result?.equity ?? rec.equity;

    if (potOdds == null || equity == null) {
        dom.potOddsIndicator.style.display = 'none';
        return;
    }

    // Only show if there's an actual decision to make involving pot odds (> 0)
    if (potOdds > 0) {
        dom.potOddsIndicator.style.display = 'block';
    } else {
        dom.potOddsIndicator.style.display = 'none';
        return;
    }

    const equityPct = Math.round(equity * 100);
    const potOddsPct = Math.round(potOdds * 100);

    dom.potOddsFill.style.width = `${equityPct}%`;
    dom.potOddsMarker.style.left = `${potOddsPct}%`;

    const isProfitable = equity >= potOdds;

    if (isProfitable) {
        dom.potOddsStatus.textContent = `Profitable Call (${equityPct}% >= ${potOddsPct}%)`;
        dom.potOddsStatus.className = 'pot-odds-indicator__status pot-odds-indicator__status--profitable';
        dom.potOddsFill.className = 'pot-odds-indicator__fill pot-odds-indicator__fill--profitable';
    } else {
        dom.potOddsStatus.textContent = `Unprofitable (${equityPct}% < ${potOddsPct}%)`;
        dom.potOddsStatus.className = 'pot-odds-indicator__status pot-odds-indicator__status--unprofitable';
        dom.potOddsFill.className = 'pot-odds-indicator__fill pot-odds-indicator__fill--unprofitable';
    }
}

function updateActionsList(actions) {
    if (!actions || actions.length === 0) {
        dom.allActions.innerHTML = '<div style="color: var(--text-muted); text-align: center; padding: 0.5rem;">Нет данных</div>';
        return;
    }

    const sorted = [...actions].sort((a, b) => b.score - a.score);
    dom.allActions.innerHTML = sorted.map(act => {
        const name = ACTION_NAMES[act.action_type] || act.action_type;
        const amt = act.amount > 0 ? ` ${act.amount.toFixed(0)}` : '';
        const pct = Math.round(act.score * 100);

        return `
            <div class="action-item">
                <span class="action-item__name">${name}${amt}</span>
                <span class="action-item__details">EV: ${act.ev >= 0 ? '+' : ''}${act.ev.toFixed(1)}</span>
                <div class="action-item__bar">
                    <div class="action-item__bar-fill" style="width: ${pct}%"></div>
                </div>
            </div>
        `;
    }).join('');
}

function updateConfidence(conf) {
    setConfidenceBar('conf-vision', conf.vision_confidence);
    setConfidenceBar('conf-ocr', conf.ocr_confidence);
    setConfidenceBar('conf-state', conf.state_confidence);
    setConfidenceBar('conf-rec', conf.recommendation_confidence);
}

function setConfidenceBar(id, value) {
    const fill = $(`#${id}`);
    const val = $(`#${id}-val`);
    const pct = Math.round(value * 100);

    fill.style.width = `${pct}%`;
    val.textContent = `${pct}%`;

    // Color coding
    if (value < 0.4) {
        fill.className = 'confidence-bar__fill confidence-bar__fill--low';
    } else {
        fill.className = 'confidence-bar__fill';
    }
}

function updateCards(ts) {
    // Hero cards
    const hero = ts.players?.find(p => p.is_hero);
    if (hero && hero.hole_cards && hero.hole_cards.length > 0) {
        dom.heroCards.innerHTML = hero.hole_cards.map(c => renderCard(c)).join('');
    } else {
        dom.heroCards.innerHTML = '<span class="card-placeholder">?</span><span class="card-placeholder">?</span>';
    }

    // Board cards
    if (ts.community_cards && ts.community_cards.length > 0) {
        dom.boardCards.innerHTML = ts.community_cards.map(c => renderCard(c)).join('');
    } else {
        dom.boardCards.innerHTML = '<span class="card-placeholder">—</span>';
    }
}

function renderCard(card) {
    const rank = card.rank === '?' ? '?' : card.rank;
    const suit = card.suit === '?' ? '' : (SUIT_SYMBOLS[card.suit] || '');
    const suitClass = SUIT_CLASSES[card.suit] || '';

    return `<span class="card-chip card-chip--${suitClass}">${rank}${suit}</span>`;
}

function updateDetections(detections) {
    dom.detectionCount.textContent = detections?.length || 0;

    if (!detections || detections.length === 0) {
        dom.debugLog.innerHTML = '<div class="debug-log__empty">Нет детекций</div>';
        return;
    }

    dom.debugLog.innerHTML = detections.map(det => {
        const conf = Math.round(det.bbox.confidence * 100);
        return `<div class="debug-log__entry">[${det.detection_class}] ${det.label || '—'} (${conf}%)</div>`;
    }).join('');
}

function showError(msg) {
    dom.explanation.querySelector('.explanation__text').textContent = `❌ ${msg}`;
}

function formatPercent(value) {
    return `${Math.round(value * 100)}%`;
}

// ─── Auto Mode ─────────────────────────────────────────────────────────────

function toggleAutoMode() {
    state.isAutoMode = !state.isAutoMode;
    dom.btnAuto.dataset.active = state.isAutoMode;

    if (state.isAutoMode) {
        dom.btnAuto.textContent = '⏹ Стоп';
        state.autoInterval = setInterval(() => {
            if (state.isConnected) {
                analyzeSynthetic();
            }
        }, 2000);
    } else {
        dom.btnAuto.textContent = '🔄 Авто';
        if (state.autoInterval) {
            clearInterval(state.autoInterval);
            state.autoInterval = null;
        }
    }
}

// ─── Init ──────────────────────────────────────────────────────────────────

function init() {
    // Event listeners
    dom.btnAnalyze.addEventListener('click', analyzeSynthetic);
    dom.btnAuto.addEventListener('click', toggleAutoMode);

    // Health check on load
    checkHealth();

    // Periodic health check
    setInterval(checkHealth, 5000);

    console.log('♠️ Poker Helper UI initialized');
}

// Start
document.addEventListener('DOMContentLoaded', init);
