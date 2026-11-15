# PacAI Capture the Flag Capstone Project
This program was my capstone project. It's a two-agent capture the flag implementation built on the `pacai` framework using `GreedyFeatureAgent` architectures and precomputed shortest-path board distances.

## Team Overview

The team is instantiated via `create_team()` and pairs two specialized feature-based agents that dynamically balance territorial defense, border control, and aggressive pellet collection:

* **`DefensiveAgent`**: Protects home territory, hunts down invading opponents, guards remaining food pellets when supplies run low, and opportunistically transitions into an attack mode when the coast is clear.
* **`OffensiveAgent`**: Crosses into enemy territory to collect food and power capsules while actively evading non-scared ghosts using a proximity-based danger heuristic.

---

## Agent Strategies & Feature Engineering

Both agents precompute maze distances at `game_start()` using `pacai.search.distance.DistancePreComputer` to evaluate successor states accurately without relying on Manhattan distance approximations.

### 1. `DefensiveAgent`
* **Invader Pursuit**: Heavily penalizes the presence of invaders (`num_invaders`: `-1000.0`) and minimizes distance to the closest intruder (`distance_to_invader`: `-40.0`). If the defender is currently scared, it overrides pursuit weights to avoid being eaten.
* **Adaptive Zone Defense**:
  * **High Food Count ($\ge 10$ pellets)**: Patrols near the center border (`keep_at_border_food_distance`) to intercept opponents early.
  * **Low Food Count ($< 10$ pellets)**: Tightens defense by hovering closer to remaining home food (`food_distance`) to prevent endgame sweeps.
* **Opportunistic Attack Mode**: When `num_invaders == 0` and no opponents are within `SAFE_ATTACK_DISTANCE` (`10.0` units), the agent drops its home-side bias and advances toward opponent food (`attack_mode`: `+20.0`).
* **Team Spacing**: Maintains separation from its teammate (`distance_to_ally`: `+2.0`) to maximize map coverage.

### 2. `OffensiveAgent`
* **Food & Score Optimization**: Prioritizes increasing the normalized team score (`score`: `+100.0`), crossing onto the opponent's side (`on_home_side`: `-30.0`), and minimizing distance to enemy pellets (`distance_to_food`: `-1.5`).
* **Capsule Targeting**: Actively seeks out opponent power capsules (`capsules`: `6.0` applied to negative distance) to turn enemy defenders into scared ghosts.
* **Threat Detection & Evasion**:
  * Ignores ghosts outside `GHOST_IGNORE_RANGE` (`2.5` units) so it can farm pellets efficiently.
  * Once a non-scared ghost enters `2.5` units while the agent is a Pacman, food priority drops and an inverse-distance danger penalty (`10.0 / max(0.5, closest_ghost_distance)` weighted at `-15.0`) forces immediate evasion.
* **Movement Smoothing**: Penalizes stopping and reversing direction (`reverse`: `-10.0`) to prevent getting stuck in oscillation loops.

---

## Project Structure & Key Components

* `create_team()`: Entry point returning the `[DefensiveAgent, OffensiveAgent]` configuration for the PacAI capture engine.
* `DefensiveAgent` / `_extract_baseline_defensive_features()`: Class definition, base weights, and state-action feature extractor for defense.
* `OffensiveAgent` / `_extract_baseline_offensive_features()`: Class definition, base weights, and state-action feature extractor for offense.
* `GHOST_IGNORE_RANGE` (`2.5`): Distance threshold controlling when the offensive agent switches from pellet collection to active ghost evasion.

---

## Usage

Run a Capture the Flag match using the `pacai` CLI by pointing one of the teams to your module path (replace `your_module_name` with the filename without `.py`):

```bash
python -m pacai.bin.capture --red your_module_name
```

To test against a specific opponent or layout:

```bash
python -m pacai.bin.capture --red your_module_name --blue pacai.agents.capture.baseline --layout defaultCapture
```

### Overriding Weights
Both agent constructors accept an optional `override_weights` dictionary (`dict[str, float]`) to tune behavior during experiments or training without modifying the base class defaults.

---

## Known Notes & Potential Improvements

* **Feature Key Alignment**: In `DefensiveAgent`, the weight dictionary defines `offense_food_distance`, whereas `_extract_baseline_defensive_features` populates `attack_food_distance`. Aligning these key names will ensure the defender actively paths toward enemy food during `attack_mode`.
* **Repeat Position Tracking**: `OffensiveAgent` defines a weight for `repeat_position` (`-15.0`), which can be wired into `_extract_baseline_offensive_features` using recent position history to further reduce oscillation near contested borders.
