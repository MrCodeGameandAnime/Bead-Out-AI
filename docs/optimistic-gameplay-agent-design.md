# Optimistic Gameplay Agent Design

## Purpose

Make the Beads Out agent play continuously from partial observations. It should rank plausible actions using local evidence, act on the best available candidate, observe the result, and retain enough evidence to improve future choices. Uncertainty lowers a candidate's rank; it does not disable the board.

Implementation and verification remain offline. Device input is sent only when a user starts the live runner with `--execute`.

## Current behavior

- `policy.choose_move` rejects the board when the current feed color is unknown or below a confidence threshold. It considers only exact raised-tile and feed-color matches and returns one move.
- `main.run_live` runs one move unless loop mode is selected, then stops on confidence, one unchanged observation, or any non-game screen.
- Feed perception currently reports an unknown current color. There is no failure-screen fixture in `root/img/`.
- Debug output saves annotated frames but not ranked alternatives, full action history, or a terminal failure bundle.

## Design

### 1. Rank local action candidates

The policy returns candidates ordered by evidence. Each candidate carries its target tile, score, confidence, dependency statuses, assumptions, and an extensible evidence context. Candidate requirements cover tile legality, tile color, visible current state, feed state, special rules, lock rules, and other mechanics. Each dependency is `KNOWN`, `LIKELY`, `UNCERTAIN`, or `UNKNOWN`.

Known depressed tiles and non-game screens are excluded. Unknown legality, color, feed, special values, and lock rules lower candidate rank without globally rejecting otherwise plausible actions. Prefer understood ordinary moves first, then moves that need more assumptions. Keep special or lock-affected actions available as fallbacks. With unknown feed, visible raised and colored tiles remain playable; hidden `?` regions remain unknown and are never assigned invented colors.

An evidence context has a mechanic identity and a flexible feature map. Initial features may include tile color, kind, legality, lock state, feed state, local geometry, and neighboring state. Records compare on the mechanic identity and equal feature values they share; new feature keys can be added without changing the ledger schema. The policy queries evidence through a context lookup instead of depending on a fixed tuple key.

### 2. Record interaction and strategic outcomes separately

Each action event stores two outcome dimensions:

- Interaction: `ACTION_ACCEPTED`, `NO_CHANGE`, `NOT_ATTEMPTED`, or `UNKNOWN`.
- Strategic: `LOCAL_PROGRESS`, `LEVEL_SUCCESS`, `LEVEL_FAILURE`, `IN_PROGRESS`, or `UNKNOWN`.

A selected tile disappearing or changing proves that the action was accepted. It does not prove strategic progress. `LOCAL_PROGRESS` requires an explicit objective/progress signal distinct from tile acceptance. Current perception may leave that signal unknown. Level success and failure are stronger strategic evidence and are stored as run outcomes; they must not be inferred from a locally accepted action.

Immediate interaction evidence can update action/legality confidence in comparable contexts. Strategic evidence is kept separate and has greater weight when ranking choices. A no-change action marks that exact candidate as tried for the stable state; it does not establish a broad mechanic rule as false.

### 3. Run continuously and adapt to each observation

When `--execute` is enabled, capture state, rank candidates, attempt the best untried candidate, capture again, record both outcome dimensions, and choose again. For an unchanged state, try the next distinct candidate and do not repeat a candidate until the state changes. Reset attempted candidates after meaningful state change.

Stop when the UI cannot be recognized well enough to interact, there is no plausible candidate, all distinct candidates for a stable state yield no change, a confirmed failure is observed, device execution is unsafe, or a completed level has no recognized safe continuation control. Unknown feed, low confidence, unreadable special values, or an unfamiliar mechanic are not stop conditions.

Completion is recorded as success. Advance only through a recognized continuation control with no cost and no ad requirement. The current completion sample does not provide such a control, so it results in a success stop.

### 4. Preserve replayable evidence and learn from outcomes

Each run is stored in `root/debug/runs/<run-id>/` with an incremental JSONL event journal and before/after frames. An event includes structured states, all ranked candidates and dependencies, the chosen tile, exact tap coordinates, timing, interaction outcome, strategic outcome, and uncertain assumptions. Terminal manifests link the recent action sequence and frames. Failure runs are marked as failure examples; unknown terminal UI is retained as an unrecognized-UI example until a representative failure screenshot is available.

A persistent JSONL evidence ledger stores schema-versioned mechanic/context features and the corresponding outcome. It can accept future feature keys, including mechanic identity, local geometry, neighbors, and newly observed state. `ACTION_ACCEPTED` contributes only interaction evidence. Explicit local progress and level outcomes contribute strategic evidence. `NO_CHANGE` remains candidate-specific and does not invalidate a general mechanic.

The first version records evidence and adjusts ranking; it does not automatically rewrite detector thresholds or mechanic rules from ambiguous outcomes.

## Interfaces and files

- `root/src/beads_bot/board.py`: shared evidence context, candidate requirements, and distinct outcome types.
- `root/src/beads_bot/policy.py`: local candidate construction, ranking, and contextual evidence lookup.
- `root/src/beads_bot/main.py`: continuous execution, per-state attempt tracking, and terminal handling.
- `root/src/beads_bot/recording.py` (new): run journal, frame storage, terminal manifests, and persistent contextual evidence.
- `root/src/beads_bot/debug.py`: candidate-ranked annotations and selected-action explanation.
- `root/tests/test_policy.py`, plus new `test_recording.py` and `test_runner.py`: deterministic offline regressions.
- `docs/README.md`: live command semantics, outcome meanings, stop conditions, and artifact locations.

Feed geometry refinement is intentionally deferred. Unknown feed is already playable. Add feed hypotheses only when a real progression failure is recorded and its evidence implicates feed ordering or outlet geometry.

## Verification

Offline regressions must establish that unknown feed still yields candidates; safer candidates rank above uncertain fallbacks; no-change tries a distinct candidate without repeating; board changes resume ranking; accepted interaction is not counted as strategic progress; explicit local progress and level outcomes are recorded distinctly; contexts with new feature keys remain readable and comparable on shared features; terminal manifests retain recent states, rankings, actions, frames, and assumptions; success does not tap a paid/ad control; and existing Level 56, 57, 59, 60, and 79 perception regressions remain passing.

Tests do not invoke ADB or send gameplay input. No live gameplay is run as part of this implementation.

## Scope and assumptions

- Existing tile colors, relief labels, and lock markers remain the local perception inputs.
- Feed inference is not a prerequisite for action and is outside this implementation.
- Evidence remains inspectable and outcome-specific; ambiguous observations do not become broad mechanic rules.
- Live device execution remains opt-in through `--execute`.
