# bloq_editor

A desktop and browser editor for building block graphs and inspecting compiled
Bloq programs.
[Open the browser editor](https://bloqec.com/editor/), or run `just editor`
for desktop and `just web` for the browser from the repository root.
See the [installation guide](https://bloqec.com/docs/dev/getting-started/installation.html)
for setup.

## Editing workflow

- **Edit** places geometry. Choose a block in the dock and click an empty grid
  cell; use **J/K** to change the placement layer. Choose **Pipe** (or press **P**)
  and click a starting block, then a highlighted adjacent endpoint. **W/A/S/D**
  and **Up/Down** extend pipes, or follow an existing connection without changing
  it. The **Hadamard** checkbox lets trackpad users place Hadamard pipes without
  holding a key; holding **H** is also supported.
- **View** selects, inspects, and moves geometry. **V** enters View, and **Space**
  switches between View and Edit. Shift-click toggles selection; drag on empty
  space to box-select, or Shift-drag to add to the selection. Select one element
  and click **Attributes** (or press **Enter**), or double-click it, to edit its
  attributes, including changing a port's block type. Shift+V translates the
  graph or selection down in View. Choosing a placement tool is different from
  editing attributes: it creates new geometry.
- **Delete/Backspace** remove the selection in View or Edit. Deleting a block
  also removes its connecting pipes and orphan ports. Undo restores the edit.
  Middle-click never deletes geometry; the middle button is reserved for camera
  orbit.
- **Actions** opens the action DAG for measurements, feedback, and other dynamic
  behavior. It is available in the toolbar even while pipe placement owns the
  **A** key.
- **BLOG** edits the active graph live in both directions. After a short typing
  pause (250 ms), background analysis applies valid text; visual edits refresh
  the text. Incomplete or invalid text stays intact and pauses viewport editing.
  **Keep graph** discards pending text after confirmation. If the graph changes meanwhile, **Use text** explicitly
  applies the text instead. Visual edits regenerate formatting and comments;
  **Text history** retains previous source versions for recovery, including after
  restarting. **Save draft** (desktop) or **Download draft** (browser) saves the
  exact text, including comments and incomplete edits. Saving a BLOG file from
  the File menu instead exports the last valid graph, not recovery history.
  `BLOG …` means text is being checked; `BLOG !` needs attention.

Desktop sessions autosave locally in the platform application-data directory
(on macOS: `~/Library/Application Support/bloq-editor/session-v1.json`), with a
previous-save backup. Browser sessions use IndexedDB with transactional saves
and a previous-save backup. Existing localStorage recovery is migrated once and
left untouched; older editor versions still write their separate legacy save.
The retained Undo/Redo stack (up to 20 snapshots), unfinished BLOG drafts and text
history survive restart. History is never silently trimmed to fit a quota.
Storage failures preserve the preceding save and are reported. Export important
work independently: browser storage can be cleared, and closing a browser or
terminating a process can interrupt a save that has not finished.

Startup recovery runs in the background. If you start working before it finishes,
saved tabs open beside your new work instead of replacing it. Native analysis
and disk writes also run in the background. Browser live analysis, BLOG imports
and recovery use independent workers; session serialization and IndexedDB writes
stay off the UI thread. Checked snapshots transfer source graphs without
repeating BLOG parsing or module geometry certification on the UI thread;
normal validation and compilation checks still apply. Comment-only edits transfer
no replacement graph. Structured copying, snapshot decoding/index construction
and scene rendering still scale with document size. Very large BLOG text buffers
can still slow text layout and painting. Failed browser saves back
off for unchanged sessions; new changes retry normally.
Live BLOG analysis reuses one idle worker to avoid repeated WebAssembly startup;
canceling an in-flight analysis still terminates it. The idle worker retains its
WebAssembly memory until the page closes.

Safari compatibility remains tracked in [issue #44](https://github.com/inmzhang/bloq/issues/44);
the editor does not patch upstream Bevy shaders.

Use **Help** for the full shortcut reference. On Mac, use Command in place of
Ctrl for application shortcuts. Camera orbit is available with **Alt + left
drag**, without a middle mouse button; **Shift + Alt + left drag** or right-drag
pans, and the wheel zooms.
