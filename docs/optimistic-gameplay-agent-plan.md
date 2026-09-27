# Optimistic Gameplay Agent Implementation Plan

**Goal:** Make the Beads Out runner choose and attempt plausible actions through uncertainty, learn from distinct interaction and strategic outcomes, and retain reproducible evidence.

**Architecture:** The policy ranks actions using local requirements and an extensible context. The recorder stores full state, candidate, action, and outcome data. The live runner continuously observes and acts when `--execute` is enabled. Feed geometry inference is deferred until a real progression failure points to feed ordering or outlet geometry.

**Tech Stack:** Python, `unittest`, Pillow, NumPy, ADB for explicitly enabled live execution. No new package dependencies.

**Spec:** [docs/optimistic-gameplay-agent-design.md](docs/optimistic-gameplay-agent-design.md)

## Global Constraints

- Live input remains opt-in through `--execute`; implementation tests use injected capture and tap functions and never invoke ADB.
- Unknown confidence, feed, special values, or mechanics lower action rank but do not globally block plausible tile actions.
- Hidden `?` conveyor regions remain unknown; this implementation does not infer feed geometry.
- Known DNH tiles and non-game screens are not action candidates. Unknown, special, and lock-affected plausible tiles remain available as lower-ranked fallbacks.
- Action acceptance and strategic progress are separate. Tile disappearance or board change alone can prove acceptance, never `LOCAL_PROGRESS` or a successful strategy.
- Evidence context is schema-versioned and extensible: mechanic identity plus a flexible feature map that can hold local geometry, neighbors, and future observed features.
- Existing Level 56, 57, 59, 60, and 79 perception regressions must continue to pass.
- Do not create a new `docs/superpowers` directory. The user-approved plan and design remain in `docs/`.

## Review Focus

1. Unknown feed still yields plausible candidates and never creates a guessed hidden value. Tested by Task 1; no feed detector work in this pass.
2. Unknown/special/locked actions are attempted as fallbacks, after stronger candidates. Tested by Task 1 and Task 3.
3. Tile removal records `ACTION_ACCEPTED`, not strategic progress. `LOCAL_PROGRESS` requires a distinct explicit progress signal. Tested by Tasks 2 and 3.
4. New evidence feature keys can be stored and matched on shared keys without changing the ledger schema. Tested by Task 2.
5. No-change actions try a different candidate for the same state; accepted actions do not stop the run. Tested by Task 3.
6. Level outcomes and completion handling stop or continue only on recognized state/control. Tested by Task 3.
7. Failure artifacts preserve the recent pre/post states, candidates, exact actions, confidence, frames, and assumptions. Tested by Task 2 and Task 3.

---

### Task 1: Model and rank local action candidates

**Files:**
- Modify: `root/src/beads_bot/board.py`
- Modify: `root/src/beads_bot/policy.py`
- Test: `root/tests/test_policy.py`

**Interfaces:**
- `EvidenceStatus = Literal["KNOWN", "LIKELY", "UNCERTAIN", "UNKNOWN"]`.
- `EvidenceContext(mechanic_id: str, features: Mapping[str, object])`; `features` contains JSON-compatible data and accepts future keys without a schema migration.
- `RequirementEvidence(name: str, status: EvidenceStatus, confidence: float, source: str)`.
- `ActionCandidate(key: str, context: EvidenceContext, tile: Tile, score: float, confidence: float, requirements: tuple[RequirementEvidence, ...], assumptions: tuple[str, ...])`.
- `EvidenceTally` keeps interaction counts and strategic counts in distinct fields.
- `rank_candidates(state, *, attempted=frozenset(), evidence_for: Callable[[EvidenceContext], EvidenceTally] | None = None)` returns all plausible candidates in rank order.
- `choose_move` remains as a compatibility wrapper and includes all ranked candidates in `Decision`.
- Context features include the currently observed tile/feed/lock classes and may include geometry or neighboring state when available. Similarity requires equal values for shared features under the same mechanic identity; extra feature keys do not break comparison.

**Steps:**

1. Replace feed-abstention tests with `test_unknown_feed_still_returns_plausible_candidates` and `test_unknown_feed_does_not_invent_a_color`.
2. Replace categorical lock rejection tests with `test_understood_unlocked_tile_ranks_above_uncertain_candidates` and `test_uncertain_special_or_locked_tile_remains_as_fallback`.
3. Add `test_known_dnh_tiles_are_not_candidates` and `test_contextual_evidence_adjusts_candidate_rank_without_removing_it`.
4. Run the focused policy tests and confirm they fail against the current early return and single exact-match policy.
5. Implement evidence types and ranking. Known DNH and non-game are hard filters; uncertain requirements change rank only. Keep unknown feed candidates eligible.
6. Run the focused policy suite and the full existing suite.

**Verification:** `$env:PYTHONPATH='root/src'; py -m unittest discover -s root/tests -p test_policy.py -v`, then `$env:PYTHONPATH='root/src'; py -m unittest discover -s root/tests -v`.

---

### Task 2: Record outcomes and extensible evidence contexts

**Files:**
- Create: `root/src/beads_bot/recording.py`
- Create: `root/tests/test_recording.py`
- Modify: `root/src/beads_bot/board.py` only for shared interaction/strategic outcome types or an explicit optional progress signal.

**Interfaces:**
- Interaction outcomes: `ACTION_ACCEPTED`, `NO_CHANGE`, `NOT_ATTEMPTED`, `UNKNOWN`.
- Strategic outcomes: `LOCAL_PROGRESS`, `LEVEL_SUCCESS`, `LEVEL_FAILURE`, `IN_PROGRESS`, `UNKNOWN`.
- Events store both outcome fields. A tile disappearance or board change may yield `ACTION_ACCEPTED`; `LOCAL_PROGRESS` requires a separately observed objective/progress value. If no such value is available, leave strategic outcome `IN_PROGRESS` or `UNKNOWN`.
- `RunRecorder.create(base_dir: Path, evidence_path: Path) -> RunRecorder`.
- `RunRecorder.record_step(...) -> str` writes full before/after states, all candidate contexts and scores, chosen action and tap, outcomes, timings, and before/after frame paths. It supports observations with no candidates or no attempted action.
- `RunRecorder.finish(status, reason, *, recent_steps=8) -> Path` writes a manifest referencing the recent action sequence and frames.
- `RunRecorder.evidence_for(context: EvidenceContext) -> EvidenceTally` reads the schema-versioned JSONL ledger. Records retain `mechanic_id`, arbitrary feature maps, outcome type, run id, and step/episode id.
- Per-action `ACTION_ACCEPTED` evidence updates interaction evidence only. No-change is recorded against the specific context and does not invalidate a mechanic globally. Level outcomes are stored as strategic episode evidence with the sequence contexts, not falsely attributed to one final tap.

**Steps:**

1. Add `test_action_event_keeps_states_candidates_tap_and_both_outcomes`, `test_failure_manifest_links_recent_action_sequence`, and `test_no_candidate_observation_is_journaled`.
2. Add `test_tile_disappearance_is_acceptance_not_local_progress`, `test_explicit_progress_signal_records_local_progress`, and `test_level_outcome_is_stored_as_episode_evidence`.
3. Add `test_context_ledger_round_trips_future_features` using mechanic identity, local geometry, neighbors, and an additional previously unknown key.
4. Run the focused recorder tests and verify expected failures before implementing `recording.py`.
5. Implement incremental event/frame recording and the generic schema-versioned evidence ledger.
6. Run recorder tests and the full suite.

**Verification:** `$env:PYTHONPATH='root/src'; py -m unittest discover -s root/tests -p test_recording.py -v`, then the full suite.

---

### Task 3: Run continuously through uncertain states

**Files:**
- Modify: `root/src/beads_bot/main.py`
- Modify: `root/src/beads_bot/board.py` only if runner state requires an optional screen control/progress field.
- Test: `root/tests/test_runner.py` (new)

**Interfaces:**
- Inject `capture_fn`, `tap_fn`, `recorder_factory`, and `sleep_fn` into `run_live` so tests are deterministic and offline.
- The runner consumes the ranked candidates and contextual evidence lookup. It has no minimum-confidence or unknown-feed stop gate.
- Attempted candidate keys are scoped to a stable state signature. An unchanged observation tries the next candidate; a state change resets the set.
- `--execute` starts continuous operation. `--max-moves` is optional and unbounded by default; `--loop` remains a backwards-compatible alias. Dry run captures and reports candidates without input.
- A free, recognized, non-ad `continue` control may be tapped after confirmed completion. A paid or ad control is never tapped automatically.
- A failure screen stops input immediately. Unknown non-game UI is saved as an `unrecognized_ui` terminal example.

**Steps:**

1. Add `test_unknown_feed_executes_best_candidate_and_observes_again`, `test_low_confidence_candidate_does_not_stop_execution`, `test_accepted_action_continues_to_next_decision`, and `test_unchanged_action_tries_next_candidate_without_repeating`.
2. Add `test_progress_resets_attempts`, `test_confirmed_failure_stops_input`, `test_unrecognized_ui_stops_and_records_frame`, `test_completion_without_safe_control_stops_without_tapping`, and `test_free_continue_resumes_game`.
3. Run runner tests and verify they fail against the current one-shot, confidence-gated loop.
4. Implement continuous observe–rank–act–observe behavior and terminal handling with injected I/O.
5. Run runner tests and the full suite. Confirm no ADB command is invoked by the tests.

**Verification:** `$env:PYTHONPATH='root/src'; py -m unittest discover -s root/tests -p test_runner.py -v`, then the full suite.

---

### Task 4: Surface ranked evidence and document live operation

**Files:**
- Modify: `root/src/beads_bot/debug.py`
- Modify: `root/src/beads_bot/main.py`
- Modify: `docs/README.md`
- Test: `root/tests/test_policy.py` and `root/tests/test_runner.py`

**Steps:**

1. Add `test_report_serializes_ranked_candidate_evidence_and_outcomes` and `test_live_summary_links_run_journal_and_terminal_status`.
2. Run the report test and confirm its candidate-list assertion fails. The live-summary path is already implemented in Task 3, so its regression may pass before Task 4; preserve that coverage instead of changing it merely to force a red result.
3. Show candidate rank, requirement states, assumptions, and selection in offline reports/annotations. Include the run journal and terminal manifest in live summaries.
4. Update the README: `--execute` is continuous; explain interaction versus strategic outcomes, terminal stop conditions, and run/evidence locations. State that feed refinement is deferred and unknown feed still permits ranked actions.
5. Run the full suite and inspect offline screenshot output. No live gameplay or taps are part of this implementation.

**Verification:** `$env:PYTHONPATH='root/src'; py -m unittest discover -s root/tests -v`.

---

## Plan self-review

- **Order:** Candidate ranking comes first, then evidence recording, then the continuous runner, then reporting. Feed inference is not a prerequisite and has no implementation task.
- **Outcome separation:** Interaction acceptance, no change, local progress, and level outcomes have distinct fields and tests. Tile disappearance does not validate strategy.
- **Context growth:** Ledger entries carry a mechanic id and extensible JSON feature map. The matching rule tolerates new keys while comparing shared context.
- **Integration:** Runner consumes Task 1 candidates and Task 2 evidence/recorder interfaces. Reporting serializes those same objects. Each task includes a red test step and a focused plus full-suite verification command.
- **Scope:** The Level 56, 57, 59, 60, and 79 vision regressions remain required. No live device input is invoked during implementation.
