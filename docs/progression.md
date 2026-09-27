`docs/PROJECT_EVOLUTION.md`

And I’d write it roughly like this:


# Bead-Out-AI: Project Evolution

## Why This Document Exists

Bead-Out-AI began as a very small experiment: automate a mobile puzzle game.

It is already evolving into something more interesting.

As the project grows, ideas like partial knowledge, optimistic execution,
failure-driven regression, active curiosity, and generalized software
interaction will become increasingly abstract.

This document preserves the path that led to those ideas.

The goal is to make it possible to look back later and understand:

- where the project started
- what problems forced the architecture to change
- what has actually been implemented
- what remains experimental
- where the project may eventually lead

This is not a specification.

It is the history and reasoning behind the system.


# 1. Where We Started

## The Original Goal

The original objective was simple:

> Play Beads Out automatically.

Beads Out is a mobile conveyor-belt puzzle game. The player observes a board,
identifies currently usable boxes, follows incoming bead colors, interacts with
special mechanics, and progresses through increasingly complex levels.

The first attempt did not involve building a dedicated AI system.

Instead, Codex Computer Use was used to visually inspect the game through
Windows Phone Link and interact with it.

This proved that a general-purpose AI model could understand the game well
enough to complete levels.

However, it also exposed a major limitation.

Codex was extremely slow.

Each move required:

1. capturing the screen
2. sending the state through a general-purpose reasoning model
3. visually interpreting the board
4. deciding on one action
5. interacting with the UI
6. observing the next state

The approach worked, but it was not practical for sustained gameplay.

More importantly, it was not really our AI.

It was someone else's general-purpose AI being instructed to play the game.


# 2. The First Architectural Shift

The project then changed direction.

Instead of asking a general-purpose model to play Beads Out, the goal became:

> Build a purpose-built local system specifically designed to perceive,
> reason about, and interact with Beads Out.

This introduced the first dedicated Bead-Out-AI architecture.

The initial implementation used:

- Python
- Pillow
- NumPy
- classical computer vision
- ADB screenshots
- ADB input
- deterministic game-state reconstruction
- rule-based action selection

No large language model was required in the gameplay loop.

The system began detecting:

- game screens
- board geometry
- individual tiles
- tile colors
- raised versus depressed tiles
- lock markers
- special tiles
- difficulty indicators
- changing board states

The objective was no longer:

> Ask an AI what to click.

It became:

> Build a system that understands enough of the environment to decide for itself.


# 3. The First Major Problem: Perception

The earliest detector worked on known screenshots but quickly failed when the
game introduced different board layouts and mechanics.

This exposed an important fact:

> The game cannot be understood from one representative level.

New levels introduced:

- different grid dimensions
- hidden beads
- lock overlays
- numbered special tiles
- different tile spacing
- different visual backgrounds
- changing conveyor layouts
- new mechanics

The detector therefore began evolving through regression.

A failure on a new level became a new test case.


## Level 59 / Level 60

Early Very Hard levels exposed weaknesses in:

- grid reconstruction
- lock detection
- special-tile recognition
- raised/depressed classification
- conveyor interpretation

The system originally attempted to infer special values such as `2`, `3`,
`5`, `200`, and `600` using weak visual heuristics.

Those guesses were unreliable.

The architecture was changed so that:

> Unknown values remain unknown.

The system should represent uncertainty rather than manufacture certainty.


## Level 79

Level 79 exposed another major perception failure.

The existing raised/depressed classifier measured too much surrounding
brightness and incorrectly classified much of the board as raised.

A new regression case was added with explicit ground truth:

Raised:

- `(174, 1457)`
- `(321, 1457)`
- `(467, 1457)`
- `(614, 1457)`
- `(760, 1457)`
- `(901, 1457)`

The remaining occupied cells were depressed.

The classifier was changed from a broad brightness measurement to a narrow
near-white halo attached directly to each tile.

Uncertain relief scores remain `UNKNOWN`.

At this stage, the test suite preserved behavior across the previously observed
Level 56, 57, 59, 60, and 79 cases.

This established the project's first important development pattern:

> Progress until something breaks.
> Capture the failure.
> Turn it into a regression.
> Fix the failure without breaking previous knowledge.
> Continue progressing.


# 4. The Second Major Problem: Excessive Conservatism

As perception improved, a larger architectural problem became obvious.

The agent had become too conservative.

It treated unresolved information as a reason to stop the entire gameplay loop.

For example:

- feed order unknown -> stop
- special value unknown -> stop
- unfamiliar mechanic -> stop
- confidence below threshold -> stop

This behavior was logically safe but operationally useless.

Beads Out continuously introduces new mechanics.

A system that requires complete understanding before acting will eventually
stop on every sufficiently novel level.


# 5. Partial Knowledge

This led to the next major principle:

> The system does not need to understand the entire environment in order to act.

Knowledge should be local.

If one mechanic is unknown, that uncertainty should affect only actions that
depend on that mechanic.

An unrelated known action should remain available.

The agent should distinguish between states such as:

- known
- likely
- uncertain
- unknown

Unknown information should reduce confidence.

It should not automatically disable the system.


# 6. Optimistic Execution

The next step extends partial knowledge further.

The project is moving away from:

> Prove that an action is safe before executing it.

Toward:

> Choose the best available action and validate it against the resulting state.

The bot is allowed to make mistakes.

A failed level is useful information.

This changes failure from an exceptional condition into part of the development
process.


## The Intended Gameplay Loop

The emerging loop is:

```text
observe
↓
reconstruct state
↓
identify candidate actions
↓
estimate confidence and assumptions
↓
choose the best available action
↓
execute
↓
observe the result
↓
update understanding
↓
continue
```

If the level succeeds, continue.

If the level fails, preserve the evidence and improve the system.


# 7. Failure-Driven Regression

Failure should produce structured evidence.

A useful failure record should eventually contain:

- screenshots before recent actions
- reconstructed state
- detected objects
- candidate actions
- confidence scores
- assumptions used by the selected action
- exact action executed
- resulting state
- failure screen
- mechanics involved
- suspected failure category

Potential categories include:

- perception failure
- state-reconstruction failure
- policy failure
- execution failure
- timing failure
- unknown mechanic
- incorrect assumption

The important principle is:

> Fix what actually breaks.

Do not build speculative support for mechanics that have never been encountered.


# 8. Brute-Force Regression Through Progression

This became the project's core development philosophy.

The game itself acts as the curriculum.

Progression exposes increasingly difficult situations.

Those situations expose assumptions in the bot.

Those broken assumptions become regression cases.

The system therefore improves by playing.

Conceptually:

```text
PLAY
↓
PROGRESS
↓
ENCOUNTER NOVELTY
↓
ATTEMPT
↓
SUCCEED OR FAIL
↓
RECORD RESULT
↓
FIX OBSERVED FAILURE
↓
ADD REGRESSION
↓
PLAY AGAIN
```

This is continuous iteration rather than complete up-front reverse engineering.


# 9. Human-in-the-Loop Operation

The immediate goal is not perfect autonomy.

The useful target is semi-autonomous progression.

The bot should handle everything it understands.

When it encounters something genuinely blocking progress, a human can take over,
demonstrate or complete the interaction, and provide a new example.

The system then incorporates that example into future behavior.

The intended relationship is:

```text
BOT
BOT
BOT
UNKNOWN MECHANIC
HUMAN
REGRESSION
BOT
BOT
BOT
```

Human intervention should become less frequent as experience accumulates.


# 10. From Uncertainty to Curiosity

A more general idea has emerged from this work.

A capable agent should not merely represent uncertainty.

It should be able to investigate uncertainty.

Instead of:

> I do not know, therefore I must stop.

The future behavior should become:

> I do not know. What action could give me useful evidence?


## Epistemic Actions

This introduces a distinction between two categories of action.

### Goal Actions

Actions intended to directly advance the task.

Examples:

- clicking a playable Beads Out tile
- saving a file
- submitting a form
- moving an object

### Information Actions

Actions intended primarily to reduce uncertainty.

Examples:

- inspect metadata
- open a properties panel
- zoom into an object
- examine another part of the interface
- search documentation
- compare against previous examples
- perform a reversible experiment
- search externally for additional evidence

These actions may not directly advance the goal.

They improve the system's understanding so that later actions can.


# 11. A Generalized Example

Imagine the future agent interacting with GIMP.

It observes an image.

It knows:

- GIMP is open
- an image exists on the canvas
- the image likely contains an animal

It does not know:

- exactly what animal it is

The system should not treat the entire screen as unknown.

Instead it could reason:

```text
Image exists: known
Animal present: likely
Bird vs chimp: uncertain
Exact species: unknown
```

If identifying the animal matters to the task, the agent could become curious.

It might:

1. inspect image metadata
2. check filename or tags
3. zoom into the subject
4. crop the relevant region
5. perform external visual search
6. search distinguishing visual characteristics

Perhaps it learns:

```text
animal
→ bird
→ yellow bird
→ likely American goldfinch
```

The important behavior is not specifically knowing how to identify birds.

The important behavior is:

> Identify what is unknown, determine whether it matters, seek useful evidence,
> update belief, and continue.


# 12. Curiosity Must Be Goal-Directed

Curiosity cannot be unlimited.

An autonomous system could waste enormous amounts of time investigating
irrelevant uncertainty.

The system therefore needs an information budget.

The governing rule should be:

> Learn only as much as the current objective requires.

If the user's goal is merely to resize the image, identifying the bird species
is irrelevant.

If the user's goal is to label the image correctly, species identification may
be necessary.

Curiosity should serve the objective rather than replace it.


# 13. Where We Are Now

At the current stage, Bead-Out-AI has demonstrated the foundation of the idea.

Implemented or substantially demonstrated:

- local screenshot capture
- device interaction
- visual board detection
- tile reconstruction
- tile color classification
- raised/depressed classification
- special-tile detection
- lock detection
- structured game state
- regression testing against real levels
- conservative representation of unknown information
- preservation of earlier behavior as new regressions are added

Currently being redesigned:

- action selection under partial knowledge
- optimistic execution
- per-action dependency reasoning
- failure capture
- continuous gameplay
- human handoff behavior

Conceptual future direction:

- confidence-aware action ranking
- experimental/probe actions
- learning from successful uncertain actions
- reusable experience storage
- active information gathering
- generalized curiosity
- transfer from game interaction to arbitrary software workflows


# 14. Where This Could Go

Bead-Out-AI may remain a game-playing experiment.

But the architecture being discovered has broader implications.

The underlying problem is not unique to Beads Out.

Many real systems require an agent to:

- interact with software it does not control
- infer state from visible output
- operate without internal APIs
- tolerate incomplete information
- encounter unexpected UI states
- make progress despite uncertainty
- recognize objective failure conditions
- diagnose its own failure points
- improve from observed examples

That describes many business workflows.

Potential environments include:

- legacy desktop software
- vendor portals
- inventory systems
- back-office workflows
- document-processing tools
- quality assurance
- repetitive operations
- systems with incomplete or changing APIs


# 15. The Larger Experiment

The long-term experiment is therefore becoming:

> Can we build a system that enters software it does not own, learns enough
> about that environment to act, objectively identifies where its assumptions
> fail, gathers evidence when necessary, and becomes more capable through
> continuous experience?

Beads Out is a useful starting environment because:

- the rules are externally controlled
- the software is not ours
- the system receives no privileged internal state
- new mechanics appear during progression
- actions have visible consequences
- success and failure are objectively observable
- experimentation is inexpensive
- regression examples are easy to collect

The game is simple.

The architectural problem is not.


# 16. Guiding Principles

The current philosophy can be summarized as:

> Observe what is actually present.

> Represent uncertainty instead of inventing certainty.

> Do not require complete understanding before acting.

> Prefer known actions, but tolerate uncertainty when necessary.

> Act toward the objective.

> Treat success as evidence.

> Treat failure as evidence.

> Preserve failures as regression cases.

> Fix observed weaknesses rather than hypothetical ones.

> Seek information when uncertainty blocks meaningful progress.

> Learn only as much as the objective requires.

> Continuously improve through interaction.


# 17. The Direction

The project began with:

> Can Codex play this mobile game?

It became:

> Can we build our own system to play this mobile game?

It is now becoming:

> Can a purpose-built agent operate effectively under partial knowledge and
> improve continuously from its own interaction history?

The longer-term question is:

> Can the same architecture learn to operate unfamiliar software, diagnose its
> own uncertainty and failure points, actively gather missing information, and
> progressively become useful in real-world workflows?

That is the path this project is currently exploring.
