1. **Make action acceptance local to the chosen tile.**\
   `ACTION_ACCEPTED` should mean the tapped tile disappeared, changed state, changed legality, or caused some other directly attributable local effect. A one-pixel bbox shift somewhere else should not count. This also means stable-state comparison should tolerate tiny detector jitter.
2. **Recognize “Out of Space!” as a real failure regression.**\
   That second run is your first genuine labeled failure episode. Add the screenshot as a fixture, classify it as `failure`, and make that episode write `LEVEL_FAILURE` instead of `UNKNOWN`.
3. **Detect the key overlay as a mechanic.**\
   The bot is currently treating those cyan tiles as ordinary candidates because the overlay is invisible to the structured state. Even if we do not yet know exactly what the key mechanic means, the overlay should become an explicit feature like `mechanic_overlay="key"` or `unknown_overlay=True` so those tiles rank below clean ordinary candidates until experience says otherwise.
4. **Only then improve strategic move selection.**\
   X is right that the first sequence was basically confidence ordering: blue → red → pink → blue → orange with no feed knowledge. That is expected from the current policy. But we need trustworthy action/outcome labels before learning better strategy from those runs.

---

**Of those four, only #2 has really been addressed, and we deliberately changed its semantics.**

1. **Action acceptance local to chosen tile: NO, not yet.**  
   The last code I can verify still classifies `ACTION_ACCEPTED` from a whole-state signature change. That is exactly what allowed detector jitter or unrelated board changes to make a repeated cyan tap look accepted. We still need chosen-tile-local change detection plus jitter tolerance.

2. **Out of Space regression: YES, but corrected.**  
   We learned that `Out of Space` is **not** `LEVEL_FAILURE`. It is now its own `out_of_space` state with a safe `close` action. The explicit `Level Failed` screen is the thing that should record `LEVEL_FAILURE`. The raw Level 80 visual variant also got incorporated into that semantic recognition.

3. **Key overlay mechanic: NO, not yet.**  
   X identified it in the Level 80 recordings, but nothing you’ve reported since says the structured tile state gained a `key`/overlay mechanic. So those cyan key-marked tiles can still potentially be treated like ordinary tiles.

4. **Strategic move intelligence: NO, not yet.**  
   We intentionally deferred this until the evidence was more trustworthy. Move selection is still fundamentally candidate-confidence ranking, not actual Beads Out strategy. Feed inference is explicitly still deferred too.

So the current state is basically:

**✅ autonomous control loop**  
**✅ partial-knowledge execution**  
**✅ logging/evidence**  
**✅ semantic failure/recovery flow**  
**✅ Out-of-Space variants**  
**❌ local action acceptance**  
**❌ key-overlay understanding**  
**❌ intelligent game strategy**

I’d resume exactly where we left off now: **#1 first, then #3, then let it play again before doing #4.**

Because if we improve strategy while `ACTION_ACCEPTED` can still lie about whether the chosen tile actually changed, we’ll be training/reinforcing policy from contaminated evidence.
