# Optimistic Gameplay Agent Implementation Plan

**Goal:** Make the Beads Out runner keep choosing plausible actions through uncertainty, learn from observed outcomes, and retain reproducible run evidence.

**Architecture:** The policy will return ranked tile candidates with per-action dependency evidence. Perception will provide ranked feed hypotheses from gameplay geometry while leaving hidden regions unknown. A continuous runner will record every action and observation, avoid retrying an unchanged action, and use an evidence ledger to rank comparable future choices.

**Tech Stack:** Python, `unittest`, Pillow, NumPy, ADB for explicitly enabled live execution. No new package dependencies.

**Spec:** [docs/optimistic-gameplay-agent-design.md](docs/optimistic-gameplay-agent-design.md)

## Global Constraints

- Live input remains opt-in through `--execute`; implementation tests never invoke ADB or send taps.
- Feed and failure-screen visual outputs are treated as calibrated only when backed by screenshot regressions.
- Hidden `?` conveyor regions remain unknown and are never assigned invented colors.
- Unknown confidence, feed, special values, or mechanics do not globally block plausible tile actions.
- Existing Level 56, 57, 59, 60, and 79 perception regressions must continue to pass.

## Review Focus

1. Unknown feed with a hidden `?` region: produce plausible tile candidates without inventing a feed color. Tested by Tasks 1 and 2.
2. Ordinary actions exhausted while only special or lock-affected candidates remain: rank and attempt the strongest remaining plausible candidate. Tested by Task 1.
3. A tap causes no meaningful change: try a distinct candidate for that stable board and never repeat the same candidate until state changes. Tested by Task 4.
4. A run reaches confirmed failure or an unrecognized screen: stop input and retain the before/after evidence and recent assumptions. Tested by Tasks 3 and 4.
5. A run reaches completion with and without a recognized continuation action: record success and advance only when the control is located. Tested by Task 4.

---

### Task 1: Model and rank local action candidates

**Files:**
- Modify: `root/src/beads_bot/board.py`
- Modify: `root/src/beads_bot/policy.py`
- Test: `root/tests/test_policy.py`

**Interfaces:**
- Produce `EvidenceStatus = Literal["KNOWN", "LIKELY", "UNCERTAIN", "UNKNOWN"]`.
- Produce `RequirementEvidence(name: str, status: EvidenceStatus, confidence: float, source: str)`.
- Produce `ActionCandidate(key: str, context_key: str, tile: Tile, score: float, confidence: float, requirements: tuple[RequirementEvidence, ...], assumptions: tuple[str, ...])`.
- Produce `rank_candidates(state: GameState, *, attempted: frozenset[str] = frozenset(), evidence_counts: Mapping[str, tuple[int, int]] | None = None) -> tuple[ActionCandidate, ...]`.
- Keep `choose_move(state, *, attempted=..., evidence_counts=...) -> Decision`; extend `Decision` with the ordered `candidates` tuple and preserve its chosen `tile`, `reason`, and `confidence` fields.
- `evidence_counts[context_key]` is `(accepted_count, unsuccessful_count)`; candidate context keys represent the selected tile's color/kind/legality class, lock overlap, and feed evidence class.

- [ ] **Step 1: Write failing policy tests**

Replace the three existing feed-abstention tests with assertions that unknown feed returns candidates and ranks them below an otherwise equivalent known feed match. Replace both tests that categorically reject lock-affected tiles with assertions that they rank below unaffected candidates and remain available as a fallback.

Add tests named:

- `test_unknown_feed_still_returns_plausible_candidates`: a game state with an RH colored tile and `feed.current is None` returns that tile as the chosen candidate.
- `test_understood_unlocked_tile_ranks_above_uncertain_candidates`: a known RH, known-color, unlocked ordinary tile ranks before a special, a lock-overlapped tile, and an `UNKNOWN`-legality tile.
- `test_uncertain_special_or_locked_tile_remains_available_as_fallback`: when those are the only plausible candidates, ranking still returns them in order.
- `test_known_dnh_tiles_are_not_candidates`: a state with only DNH tiles returns no tile candidates.
- `test_comparable_outcome_evidence_adjusts_candidate_score`: reranking one candidate with accepted evidence for its context raises its score over its baseline; unsuccessful evidence lowers it without deleting it.

- [ ] **Step 2: Run the policy tests and verify the intended failures**

Run: `$env:PYTHONPATH='root/src'; py -m unittest discover -s root/tests -p test_policy.py -v`

Expected: the new behaviors fail against the current unknown-feed early return and single exact-match selection.

- [ ] **Step 3: Implement candidate evidence types and ranking in `board.py` and `policy.py`**

Rank candidates lexicographically by dependency support before using numeric score as the within-tier confidence. Include unknown feed and special/lock dependencies in `requirements`; do not use a minimum confidence threshold to remove candidates. Keep only non-game state and known DNH as hard filters.

- [ ] **Step 4: Run policy tests and the full current suite**

Run: `$env:PYTHONPATH='root/src'; py -m unittest discover -s root/tests -p test_policy.py -v`, then `$env:PYTHONPATH='root/src'; py -m unittest discover -s root/tests -v`.

Expected: all policy tests and all existing tests pass.

- [ ] **Step 5: Commit Task 1**

Commit message: `feat: rank gameplay actions by local evidence`.

---

### Task 2: Add geometry-based feed hypotheses

**Files:**
- Modify: `root/src/beads_bot/board.py`
- Modify: `root/src/beads_bot/vision.py`
- Test: `root/tests/test_vision.py`

**Interfaces:**
- Produce `FeedHypothesis(color: str, confidence: float, source: str)`.
- Extend `FeedObservation` with `hypotheses: tuple[FeedHypothesis, ...] = ()`; retain `current`, `upcoming`, `confidence`, and `source` for existing callers.
- Change `_read_feed` to `_read_feed(rgb: np.ndarray, hsv: np.ndarray, board_region: tuple[int, int, int, int] | None) -> FeedObservation`.
- `FeedObservation.current` may remain `None` when no hypothesis is calibrated; policy consumes the ranked hypotheses independently.

- [ ] **Step 1: Write failing feed tests**

Add tests named:

- `test_feed_hypothesis_uses_center_exit_geometry`: a synthetic gameplay image with a visible bead at the configured center exit produces a matching color hypothesis whose source identifies that geometry.
- `test_hidden_feed_question_mark_remains_unknown`: the Level 57 hidden conveyor and Level 60 regression frame retain `None` for hidden upcoming slots even when other visible feed hypotheses exist.
- `test_feed_hypothesis_does_not_sample_top_right_screen_edge`: a synthetic image with a colored patch only at the top right produces no feed hypothesis from that patch.
- `test_feed_hypotheses_are_ranked_and_bounded`: visible candidate colors are returned by descending confidence with each confidence in `[0.0, 1.0]`.

- [ ] **Step 2: Run the focused vision tests and verify the new assertions fail**

Run: `$env:PYTHONPATH='root/src'; py -m unittest discover -s root/tests -p test_vision.py -v`

Expected: the feed tests fail because `_read_feed` currently always returns no color or hypotheses.

- [ ] **Step 3: Implement feed hypotheses from path and center-state evidence**

Use the gameplay outlet and immediately connected visible bead sequence. Return no color for occluded or `?` regions. Do not use a fixed top-edge screen crop as a feed source. Preserve hypotheses even when `current` remains unknown.

- [ ] **Step 4: Run focused and full perception tests**

Run the focused `test_vision.py` command and then `$env:PYTHONPATH='root/src'; py -m unittest discover -s root/tests -v`.

Expected: feed geometry tests pass and all Levels 56, 57, 59, 60, and 79 regressions remain green.

- [ ] **Step 5: Commit Task 2**

Commit message: `feat: estimate feed candidates from gameplay geometry`.

---

### Task 3: Persist action, frame, and mechanic evidence

**Files:**
- Create: `root/src/beads_bot/recording.py`
- Create: `root/tests/test_recording.py`
- Modify: `root/src/beads_bot/board.py` only if a shared outcome type is required by the recorder interface.

**Interfaces:**
- Produce `RunRecorder.create(base_dir: Path, evidence_path: Path) -> RunRecorder`.
- Produce `RunRecorder.record_step(*, before_state: GameState, before_frame: Image.Image, candidates: tuple[ActionCandidate, ...], chosen: ActionCandidate | None, tap: tuple[int, int] | None, after_state: GameState | None, after_frame: Image.Image | None, outcome: str, timings_ms: Mapping[str, float]) -> str`, returning the step id. This records no-candidate observations as well as executed actions.
- Produce `RunRecorder.finish(status: str, reason: str, *, recent_steps: int = 8) -> Path`, writing a terminal manifest and returning its path.
- Produce `RunRecorder.evidence_counts() -> Mapping[str, tuple[int, int]]` for policy ranking from the persisted contextual ledger.
- Store run events under `root/debug/runs/<run-id>/events.jsonl`, before/after frames under that run's `frames/`, and the persistent evidence ledger at `root/debug/mechanic_evidence.jsonl`.
- Append evidence only when the selected tile has action-specific visible progress. Record no-change and ambiguous transitions without treating them as general mechanic failures.

- [ ] **Step 1: Write failing recorder tests**

Add tests named:

- `test_action_event_keeps_full_states_candidates_and_exact_tap`: assert before/after states, all candidate requirements and scores, chosen tile, tap coordinates, outcome, and frame paths are serialized.
- `test_failure_manifest_links_recent_action_sequence`: after more than eight actions, assert the failure manifest points to the most recent eight step ids and preserves the full event journal.
- `test_no_candidate_observation_is_journaled`: an observation with no plausible candidate writes its frame, state, empty ranking, and terminal reason.
- `test_uncertain_progress_is_appended_to_evidence_ledger`: an uncertain candidate followed by selected-tile removal writes a contextual accepted-evidence record.
- `test_no_change_records_contextual_unsuccessful_attempt`: an unchanged transition remains in the run journal and records an unsuccessful attempt for its context without marking a general rule invalid.

- [ ] **Step 2: Run recorder tests and verify the expected failures**

Run: `$env:PYTHONPATH='root/src'; py -m unittest discover -s root/tests -p test_recording.py -v`

Expected: failure because `recording.py` and `RunRecorder` do not exist yet.

- [ ] **Step 3: Implement `RunRecorder` and the contextual evidence ledger**

Write one JSONL record per decision/action, compress saved frames as JPEG, and include recent assumption summaries in terminal manifests. Make writes incremental so interruption does not discard prior steps.

- [ ] **Step 4: Run recorder tests and the full suite**

Run the focused recorder command and `$env:PYTHONPATH='root/src'; py -m unittest discover -s root/tests -v`.

Expected: tests confirm replayable action evidence and keep ambiguous no-change evidence local to its candidate context.

- [ ] **Step 5: Commit Task 3**

Commit message: `feat: record gameplay outcomes and evidence`.

---

### Task 4: Make live execution continuous and outcome-driven

**Files:**
- Modify: `root/src/beads_bot/board.py`
- Modify: `root/src/beads_bot/main.py`
- Modify: `root/src/beads_bot/vision.py`
- Test: `root/tests/test_runner.py` (new)
- Test: `root/tests/test_vision.py`

**Interfaces:**
- Change `run_live` to `run_live(args: argparse.Namespace, *, capture_fn: Callable = capture_frame, tap_fn: Callable = tap, recorder_factory: Callable = RunRecorder.create) -> dict` so offline tests can inject deterministic observations and input. The recorder factory receives the run base directory and evidence-ledger path.
- Add `ScreenControl(kind: str, center: tuple[int, int], confidence: float, cost: int | None = None, requires_ad: bool = False)` and `GameState.controls: tuple[ScreenControl, ...] = ()`. Only a recognized `continue` control with no cost and no ad requirement may advance the level.
- Add `_state_signature(state: GameState) -> tuple` containing screen, feed hypotheses, tile centers/colors/legalities/lock state/special values, controls, and lock geometry.
- Load `RunRecorder.evidence_counts()` and pass those counts to `rank_candidates` on each observation.
- Add explicit `failure` to screen outcomes. An unrecognized non-game screen after an action is terminal `unrecognized_ui` and is saved as a possible failure example until a failure screenshot regression is available.
- Change `--max-moves` default to `None`; retain an optional user-set cap and accept `--loop` as a backwards-compatible alias. `--execute` starts the continuous loop.

- [ ] **Step 1: Write failing runner tests with injected capture and tap functions**

Add tests named:

- `test_unknown_feed_executes_best_candidate_and_observes_again`: assert the highest-ranked candidate is tapped even with `feed.current is None`, and a fresh frame is captured afterward.
- `test_unchanged_action_tries_next_candidate_without_repeating`: return the same state after the first tap, then assert the next distinct candidate is tapped once and the first is not repeated.
- `test_progress_resets_attempts_and_continues`: a changed board allows a newly available candidate and another capture/tap cycle.
- `test_confirmed_failure_stops_input_and_finalizes_failure_bundle`: after a failure screen, assert no subsequent tap and a failure manifest is written.
- `test_unrecognized_ui_stops_and_records_possible_failure`: after a non-game unknown state, assert input stops and the frame/event are retained.
- `test_completion_only_advances_when_control_is_recognized`: without a recognized continuation target, record success and issue no guessed tap.
- `test_free_continue_control_advances_and_resumes_game`: with an explicitly recognized, free, non-ad continue control, tap its center and resume observation.
- `test_completion_never_taps_paid_or_ad_controls`: the Level Complete screenshot results in a success stop without tapping the `Next 25` purchase or ad-reward button.
- `test_low_candidate_confidence_does_not_stop_execution`: a plausible low-confidence candidate is tapped.

- [ ] **Step 2: Run runner tests and verify the expected failures**

Run: `$env:PYTHONPATH='root/src'; py -m unittest discover -s root/tests -p test_runner.py -v`

Expected: import or assertion failures expose the current one-shot, confidence-gated loop.

- [ ] **Step 3: Implement continuous execution and per-state action tracking**

For each stable state, pass tried candidate keys and the recorder's evidence counts to `rank_candidates`. After a no-change observation, record the event and select the next candidate. Reset tried keys after meaningful state progress. On completion, tap only a recognized free, non-ad continuation control; the current completion screenshot's paid `Next 25` and ad reward are not automatic actions. Stop only for the spec's terminal conditions or the optional user-set move cap. Record a frame before any terminal stop.

- [ ] **Step 4: Run runner tests and the full suite**

Run the focused runner command and `$env:PYTHONPATH='root/src'; py -m unittest discover -s root/tests -v`.

Expected: injected tests prove continuity and terminal behavior without calling ADB or sending taps.

- [ ] **Step 5: Commit Task 4**

Commit message: `feat: continue gameplay through uncertain states`.

---

### Task 5: Expose candidate evidence and document live operation

**Files:**
- Modify: `root/src/beads_bot/debug.py`
- Modify: `root/src/beads_bot/main.py`
- Modify: `docs/README.md`
- Test: `root/tests/test_policy.py`, `root/tests/test_runner.py`

**Interfaces:**
- Offline and live reports serialize every candidate's key, score, confidence, requirements, assumptions, and chosen action.
- Debug annotations identify the chosen candidate and visually distinguish lower-ranked alternatives without obscuring tile legality.
- CLI output returns the terminal run summary and run directory; detailed action history remains in `events.jsonl`.

- [ ] **Step 1: Write failing report and CLI tests**

Add tests named `test_report_serializes_ranked_candidate_evidence` and `test_live_summary_links_run_journal_and_terminal_status`. Assert reports include all candidate dependencies and the recorder's terminal path.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `$env:PYTHONPATH='root/src'; py -m unittest discover -s root/tests -p test_runner.py -v`.

Expected: candidate lists and run artifact paths are absent from current reports.

- [ ] **Step 3: Implement report serialization, annotation, and README updates**

Document that `--live --execute` runs continuously, explain the five terminal outcomes, show where action frames and evidence are stored, and retain a command for offline screenshot analysis.

- [ ] **Step 4: Run the full suite and inspect generated offline output**

Run: `$env:PYTHONPATH='root/src'; py -m unittest discover -s root/tests -v`; analyze existing screenshots in offline mode and verify reports include candidate rankings without touching a device.

Expected: all existing and new offline tests pass; offline analysis reports no input sent.

- [ ] **Step 5: Commit Task 5**

Commit message: `docs: explain continuous gameplay and run evidence`.

---

## Plan self-review

- **Spec coverage:** Local candidate evidence and fallback are covered by Task 1; geometry-based feed hypotheses by Task 2; persistent action and success evidence by Task 3; continuous execution, failure/no-progress handling, and completion by Task 4; surfaced explanations and CLI behavior by Task 5.
- **Step scan:** Each task has a failing test, an explicit command, a concrete production interface, a passing verification, and a commit.
- **Type consistency:** `FeedHypothesis` extends `FeedObservation`; `ActionCandidate` references `Tile`; `rank_candidates` returns candidates consumed by `Decision`, `RunRecorder`, and `run_live`; `ScreenControl` is stored on `GameState` and consumed by the runner; the injected recorder factory returns `RunRecorder`.
- **Review focus:** Each of the five listed edge conditions has named test coverage in its owning task.
- **Proportion:** Five tasks follow the five collaborating units in the approved design; the runner depends on policy and recorder interfaces, and later report work consumes those same interfaces.
