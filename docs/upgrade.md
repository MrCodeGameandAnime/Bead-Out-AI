# Upgrade Path

1. ~~**Make action acceptance local to the chosen tile.**~~

   ~~`ACTION_ACCEPTED` should mean the tapped tile disappeared, changed state, changed legality, or caused some other directly attributable local effect. A one-pixel bbox shift somewhere else should not count. This also means stable-state comparison should tolerate tiny detector jitter.~~

2. ~~**Recognize "Out of Space!" as a real failure regression.**~~

   ~~This was corrected semantically during implementation. `Out of Space` is recoverable; explicit `Level Failed` is terminal failure.~~

3. **Detect the key overlay as a mechanic.**

   The bot is currently treating those cyan tiles as ordinary candidates because the overlay is invisible to the structured state. Even if we do not yet know exactly what the key mechanic means, the overlay should become an explicit feature like `mechanic_overlay="key"` or `unknown_overlay=True` so those tiles rank below clean ordinary candidates until experience says otherwise.

4. **Only then improve strategic move selection.**

   The policy is now feed-aware, but it is still fundamentally greedy/myopic rather than planning around capacity, future options, and dead-end risk.

---

**Of those four, only #2 has really been addressed, and we deliberately changed its semantics.**

1. **Action acceptance local to chosen tile: YES.**

   Implemented as `tile-local-v2`. Acceptance now requires direct chosen-tile evidence. Neighbor-only changes remain diagnostic and produce `NO_CHANGE`. Retry identity is jitter-tolerant, and older acceptance evidence is isolated.

2. **Out of Space regression: YES, corrected semantics.**

   `Out of Space` is a recoverable state. Explicit `Level Failed` records `LEVEL_FAILURE`. Delayed terminal states at the move-limit boundary are also handled.

3. **Key overlay mechanic: NO, not yet.**

   Still the next focused upgrade.

4. **Strategic move intelligence: NO, not yet.**

   Feed inference is now implemented and influences policy, but the agent still lacks actual multi-step capacity/dead-end planning.

## Current State:

- **✅ autonomous control loop**
- **✅ partial-knowledge execution**
- **✅ logging/evidence**
- **✅ semantic failure/recovery flow**
- **✅ Out-of-Space variants**
- **✅ feed inference and temporal tracking**
- **✅ tile-local action acceptance v2**
- **✅ retry suppression**
- **✖️ key-overlay understanding**
- **✖️ intelligent game strategy**
