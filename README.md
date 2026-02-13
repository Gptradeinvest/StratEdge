# StratEdge

**Quantitative Trading Pipeline — GLD Focus**

Pipeline industrielle de génération de signaux de trading basée sur les méthodologies de Marcos López de Prado (*Advances in Financial Machine Learning*, 2018). Système mono-actif conçu pour GLD (SPDR Gold Shares), intégrant 5 stratégies, sélection automatique walk-forward, et gate de validation combinatorielle.

---

## Architecture

```
GLD_OHLCV → Dollar Bars → CUSUM Filter → Triple Barrier Labeling
    → Fractional Differentiation → Feature Engineering
    → Sequential Bootstrap → Purged Walk-Forward CV
    → Meta-Labeling → MDA Feature Pruning → CSCV PBO Gate
    → Bet Sizing → Backtest (net de coûts) → Dashboard
```

## Fonctionnalités AFML

| Module | Chapitre | Description |
|---|---|---|
| Dollar Bars | Ch.2 | Barres à volume constant en dollars, seuil auto-calibré |
| CUSUM Filter | Ch.2.5 | Filtrage d'événements par seuil de variation cumulée |
| Triple Barrier | Ch.3 | Labeling directional avec take-profit, stop-loss et timeout signé |
| Fractional Differentiation | Ch.5 | FFD (Fixed-Width Window) — stationnarité avec mémoire |
| Sequential Bootstrap | Ch.4.5 | Rééchantillonnage IID par unicité conditionnelle |
| Meta-Labeling | Ch.3.6 | Couche binaire (trade/no-trade) sur signal primaire |
| Purged K-Fold CV | Ch.7 | Validation croisée avec purge temporel et embargo |
| MDA Feature Importance | Ch.8 | Mean Decrease Accuracy, RF `max_features=1` |
| Bet Sizing | Ch.10 | Taille de position par probabilité calibrée (2×P-1) |
| CSCV PBO Gate | Ch.11 | Combinatorial Symmetric CV — Probability of Backtest Overfitting |
| SADF | Ch.17 | Supremum ADF — détection de bulles, lookback complet |
| Entropy | Ch.18 | Shannon, plug-in, Lempel-Ziv — mesure informationnelle |

## Stratégies

5 stratégies disponibles, sélectionnées automatiquement par walk-forward OOS 50/50 corrigé DSR :

| Stratégie | Signal | Barrières (TP/SL/H) |
|---|---|---|
| `momentum` | Return sur lookback > 0 → LONG | 1.0 / 1.0 / 1.0 |
| `trend` | Prix > SMA(63) → LONG | 1.0 / 1.0 / 1.0 |
| `mr` | Z-score < -1 → LONG, > +1 → SHORT | 0.8 / 0.8 / 0.8 |
| `bo` | Breakout canal Donchian | 1.0 / 1.0 / 1.0 |
| `bb` | Touche bande Bollinger inf → LONG | 0.6 / 0.6 / 0.7 |

**Sélection automatique** : score = Sharpe × (0.5 + WinRate), minimum 20 trades OOS, correction DSR (n_trials=5). Fallback → `momentum`.

## Features

11 features extraites pour le modèle RF :

- `Close_FracDiff` — série différenciée fractionnaire (d=0.4)
- `mom_ret` — return sur lookback
- `signal_strength` — intensité du signal primaire
- `vol` — volatilité réalisée
- `entropy_shannon`, `entropy_plugin`, `entropy_lz` — entropie (Ch.18)
- `sadf` — Supremum ADF (Ch.17)
- `bb_pct_b` — position relative dans les bandes de Bollinger
- `bb_bandwidth` — largeur normalisée des bandes

Pruning MDA : seules les features à importance > médiane sont conservées.

## CSCV PBO Gate

Avant toute mise en production, le pipeline valide le signal via Combinatorial Symmetric Cross-Validation (Ch.11) :

- Split 6/2 combinatoriel sur les données d'entraînement
- Calcul du PBO par méthode logit rank (`rankdata` + logit(w̄))
- Bet scaling adaptatif :
  - PBO ≤ 50% → `bet_scale = 1.0` (PASS)
  - 50% < PBO ≤ 75% → `bet_scale = 0.5`
  - PBO > 75% → `bet_scale = 0.0` (BLOCKED)

## Installation

```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

## Usage

```bash
# Signal + backtest + dashboard (GLD par défaut)
python pipe.py

# Forcer une stratégie
python pipe.py --strategy trend

# Comparer les 5 stratégies et lancer la meilleure
python pipe.py --strategy-select

# Autre ticker
python pipe.py --fetch SPY

# Mode recherche (CPCV, MDA, rapports)
python pipe.py data.csv --research

# Random search paramètres
python pipe.py --optimize --n-iter 100

# Mode cron incrémental
python pipe.py --daily

# Forcer time bars (pas de dollar bars)
python pipe.py --raw

# Timeframes multiples
python pipe.py --tf d1 h4 w1
```

### Paramètres avancés

| Argument | Défaut | Description |
|---|---|---|
| `--tf` | `d1` | Timeframe : `h4`, `d1`, `w1` |
| `--strategy` | auto | `momentum`, `trend`, `mr`, `bo`, `bb` |
| `--frac-d` | 0.4 | Ordre de différentiation fractionnaire |
| `--horizon` | 10 (d1) | Barres max triple barrière |
| `--upper` | 1.0 | Multiplicateur take-profit |
| `--lower` | 1.0 | Multiplicateur stop-loss |
| `--n-splits` | 5 | Folds purged CV |
| `--embargo` | 0.01 | % embargo entre folds |
| `--spread` | 0.04% | Spread bid-ask |
| `--swap` | 0.015% | Swap overnight / jour |
| `--start` | 2020-01-01 | Date de début fetch |

## Outputs

```
GLD_D1/
├── GLD_D1.csv                  # OHLCV source
├── signals_history.csv         # Historique complet des signaux
├── signal_latest.json          # Dernier signal (date, side, confidence, SL/TP)
├── signal_meta.json            # Métriques pipeline + backtest + PBO
├── strategy_comparison.json    # Comparaison 5 stratégies (scores OOS)
├── backtest_trades.csv         # Trades exécutés (P&L net)
└── dashboard.html              # Dashboard interactif (equity, DD, signals)
```

### signal_latest.json

```json
{
    "date": "2026-02-12",
    "side": 0,
    "side_label": "FLAT",
    "confidence": 0.4648,
    "bet_size": 0.0,
    "price": 451.39,
    "stop_loss": 435.26,
    "take_profit": 467.52,
    "regime": "BUBBLE",
    "sadf": 1.3665
}
```

## Résultats GLD (2020–2026)

| Métrique | Valeur |
|---|---|
| Stratégie sélectionnée | Trend (SMA 63) |
| Return | +9.2% |
| Sharpe | 3.80 |
| Max Drawdown | -2.11% |
| Calmar | 4.36 |
| Win Rate | 71.4% |
| Profit Factor | 1.83 |
| Trades | 91 |
| Avg Days Held | 3.1 |
| PBO | 0.0% |
| Bet Scale | 1.0 (PASS) |

### Comparaison stratégies (OOS 50/50)

| Stratégie | Sharpe | Return | Max DD | WR | Score |
|---|---|---|---|---|---|
| **trend** | **6.12** | **+4.33%** | **-0.57%** | **75.0%** | **7.65** |
| momentum | 4.52 | +3.54% | -1.15% | 65.9% | 5.24 |
| mr | 4.50 | +4.06% | -1.34% | 53.1% | 4.65 |
| bo | 5.89 | +1.61% | -0.67% | 78.6% | < 20 trades |
| bb | 1.22 | +0.31% | -1.86% | 33.3% | < 20 trades |

## Pipeline interne

```
1. Fetch OHLCV (Yahoo Finance)
2. Dollar Bars (seuil auto-calibré sur volume médian)
3. Volume Quality Check (CV, % zéros → fallback time bars si POOR)
4. CUSUM Filter → événements
5. Triple Barrier Labeling (timeout = sign(ret), AFML Ch.3.2)
6. Fractional Differentiation FFD (d=0.4)
7. Feature Engineering :
   - Momentum return, signal strength, volatilité
   - Entropy (Shannon, plug-in, LZ)
   - SADF (lookback complet, AFML Ch.17)
   - Bollinger %B + bandwidth
8. Walk-Forward (retrain every 20 bars) :
   - Sequential Bootstrap (AFML Ch.4.5)
   - RF (max_features=1, AFML Ch.8)
   - Meta-labeling + bet sizing
9. MDA Feature Pruning (> médiane)
10. CSCV PBO Gate (logit rank, AFML Ch.11)
11. Backtest net (spread + swap + slippage)
12. Dashboard HTML + exports JSON/CSV
```

## Conformité AFML

4 points de conformité stricte par rapport au texte original :

1. **Triple Barrier Timeout** (Ch.3.2) — `label = sign(return)` à expiration, jamais 0
2. **CSCV PBO Logit Rank** (Ch.11) — rank normalisé + transformation logit, pas de pairing séquentiel
3. **MDA max_features=1** (Ch.8) — toutes les instances RF utilisent `max_features=1`
4. **SADF Full Lookback** (Ch.17) — fenêtre rétrospective complète `range(0, t - min_window + 1)`

## Structure

```
pipe.py          # Pipeline monolithique (2099 lignes, 35 fonctions)
```

Single-file by design. Pas de dépendances externes au-delà du stack scientifique standard.

## Dépendances

- Python ≥ 3.9
- numpy
- pandas
- matplotlib
- scikit-learn
- scipy

## Licence

MIT

## Auteur

Gaëtan PRUVOT — 2026
