# Optimistic Gameplay Agent Design

## Purpose

Make the Beads Out agent play continuously from partial observations. The agent ranks each plausible action by its local evidence, attempts the best candidate, observes the result, and retains enough evidence to improve later decisions. Uncertainty lowers action rank; it does not disable the whole board.

Implementation and verification will remain offline. Live input is sent only when a user starts the live runner with its execution option.

## Current behavior

- `policy.choose_move` rejects the board whenever feed color is missing or below a confidence threshold. It considers only exact raised-tile and feed-color matches and returns one move.
- `main.run_live` executes one move unless loop mode is selected, then stops on a confidence threshold, one unchanged observation, or any non-game screen.
- The feed detector currently returns no color hypotheses. The screen detector distinguishes game, completion, and unknown screens; there is no failure-screen fixture in `root/img/`.
- Debug output saves annotated frames but not ranked alternatives, complete per-action state history, or a terminal failure bundle.

## Design

### 1. Ranked local action candidates

Replace the single-move policy result with an ordered set of candidate actions. Each candidate describes its tile target, score, confidence, evidence, and assumptions. Evidence is tracked independently for:

- tile legality;
- tile color;
- visible current state;
- feed state;
- special-tile rule;
- lock rule;
- other observed mechanics.

Each dependency uses `KNOWN`, `LIKELY`, `UNCERTAIN`, or `UNKNOWN`. Candidate scoring favors understood ordinary moves, stronger visible feed evidence, fewer unresolved mechanics, and actions isolated from locks. Specials, lock-adjacent tiles, uncertain legality, and unknown feed remain candidates with lower scores. A tile classified `DNH` is not a plausible tile action; `UNKNOWN` legality remains eligible at a reduced score. There is no global minimum-confidence rejection.

When feed color is known or estimated, matching actions rank higher. When feed is wholly unknown, visible raised, colored, unlocked tiles still form candidates. Hidden `?` conveyor regions remain unknown and are never assigned invented colors.

The policy exposes all candidates so the runner and journal can explain the choice. It also chooses the highest-ranked candidate for compatibility with the current execution flow.

### 2. Continuous observe–act runner

Starting live execution begins the continuous loop. After each tap the runner captures a fresh frame, reconstructs state, records the outcome, and ranks the next action. The current explicit execution option remains the switch that permits device input; once enabled, uncertainty and low confidence do not stop the run.

For an unchanged board, mark the selected action as attempted for that exact state and consider the next distinct candidate. Do not repeat the same action against the same unchanged state. If all plausible candidates are exhausted without meaningful progress, stop and record the no-progress outcome. A changed board resets the attempted-action set.

Stop conditions are:

1. the UI cannot be recognized well enough to interact;
2. no plausible candidate can be formed;
3. all distinct candidates for a stable state have produced no meaningful change;
4. a confirmed failure screen is observed;
5. capture, input, or device state makes execution unsafe;
6. a level completes and no recognized continuation action is available.

Low confidence, unknown feed, unreadable special values, and unfamiliar mechanics are not stop conditions. A confirmed completion is recorded as success. If a continuation control is recognized, tap it and resume the loop; otherwise retain the successful result and stop at that screen. The current code recognizes completion but has no continuation-control locator.

### 3. Run records and outcome evidence

Create a run directory at `root/debug/runs/<run-id>/`. Append one JSONL event for every decision and action. Events contain the full before and after structured states, ranked candidate scores and dependency statuses, exact selected tile and tap coordinates, timing, and observed outcome. Save before/after frames for each attempted action so a run can be replayed visually.

Mark terminal runs as `success`, `failure`, `no_progress`, `unrecognized_ui`, or `device_error`. For failures, write a manifest that identifies the recent action sequence, frames, pre-action states, selected actions, confidence scores, and uncertain assumptions. Preserve all action events in the run journal; the manifest links to the most useful recent sequence.

Record a successful uncertain action when the resulting state provides action-specific evidence, such as the selected tile disappearing or visibly changing. Append that evidence to a persistent JSONL ledger with its context and assumptions. Future ranking may use positive or negative evidence only for comparable contexts. A single unchanged observation is recorded as an unsuccessful attempt, not as proof that a general mechanic is invalid.

The first version uses empirical evidence to rank comparable candidates. It does not automatically rewrite detector thresholds or broad mechanic rules from ambiguous outcomes.

### 4. Outcome recognition and level advancement

Represent `game`, `complete`, `failure`, and `unknown` as explicit outcomes. Failure stops input immediately and finalizes the failure bundle. A screenshot labeled `failure` must be backed by a visual regression when the first representative frame is available. Until a failure screen is recognized, an unrecognized UI is handled by the existing allowed stop condition and recorded with its frame and state.

Completion advances only through a continuation control that the screen detector can locate. It must not tap a guessed location. The current completion sample verifies success recognition; it does not provide a continuation locator.

## Interfaces and files

- `root/src/beads_bot/policy.py`: evidence levels, candidate construction, scoring, and ranking.
- `root/src/beads_bot/board.py`: feed hypotheses and candidate or outcome data needed across vision, policy, and runner boundaries.
- `root/src/beads_bot/vision.py`: visible feed hypotheses and explicit screen outcome recognition while preserving hidden conveyor regions as unknown.
- `root/src/beads_bot/main.py`: continuous state machine, action-attempt tracking, progression, and terminal handling.
- `root/src/beads_bot/recording.py` (new): run journal, frame storage, terminal manifests, and persistent action-specific evidence.
- `root/src/beads_bot/debug.py`: candidate-ranked annotations that identify the chosen action and its evidence.
- `root/tests/test_policy.py`, `root/tests/test_vision.py`, and new runner/recording tests: deterministic offline regression coverage.
- `docs/README.md`: live command semantics, stop conditions, run artifacts, and feed/evidence behavior.

## Verification

Offline tests must demonstrate that:

- unknown feed still produces ranked plausible candidates;
- known, low-assumption candidates rank above uncertain candidates;
- uncertain candidates are selected when stronger candidates are exhausted;
- failed/no-change actions lead to alternatives without repeating the same action on a stable state;
- state progress resets per-state attempt tracking and the loop chooses again;
- success, failure, unrecognized UI, and device errors finalize the correct run status;
- every action event retains its states, candidates, chosen action, and result;
- uncertain progress is persisted as evidence and ambiguous no-change is not generalized;
- existing Levels 56, 57, 59, 60, and 79 perception regressions continue to pass.

No implementation test invokes ADB or sends gameplay input. Feed estimation and failure-screen recognition must have screenshot-backed regressions before their visual outputs are treated as calibrated.

## Scope and assumptions

- Existing tile colors, relief labels, and lock markers remain the initial local evidence sources; this design changes how policy ranks and uses them.
- Feed inference uses visible path geometry and center-state evidence, not a fixed screen-edge crop. Hidden regions remain unknown.
- Learning is limited to recorded action outcomes and ranking adjustments for comparable contexts. A future broader mechanic learner requires validated failure and success examples.
- The device runner remains opt-in. No gameplay is run as part of implementing this design.
