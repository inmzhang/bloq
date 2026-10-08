# CLI Reference

The `bloq` executable turns BLOG source into physical IR or Stim circuits. It also
converts, validates, and inspects saved IR and renders source and dependency
graphs. Native installations and the Python package use the same command runner. See
[installation](../getting-started/installation.md#cli) for package options.

Use `bloq --help` or `bloq <command> --help` for the options supported by your
installed version. Compilation starts from BLOG; `emit`, `stats`, and `validate`
start from a saved program and do not compile the source again.

## Commands and source selection

```text
bloq [OPTIONS] [INPUT]       # shortcut for bloq compile
bloq compile [OPTIONS] [INPUT]
bloq emit [OPTIONS] <INPUT>
bloq validate <INPUT>
bloq stats <INPUT>
bloq view [OPTIONS] [INPUT]
bloq gallery
bloq completion <SHELL>
```

`compile` accepts exactly one BLOG path or `--gallery ID`. `view` accepts the same
choices and also supports saved IR. Run `bloq gallery` to list the installed
entries and categories.

`-v, --verbose` adds compilation diagnostics; `-q, --quiet` suppresses status
output. They conflict with each other. Place them after a named subcommand, for
example `bloq compile --quiet input.blog -d 3`. Quiet mode retains errors and
requested artifact or inspection output. `bloq --version` prints the version.

Exit status is **0** for success, **1** for command failure, and **2** for argument
parsing errors. Help and a reader closing an output pipe also return 0. A parsed
argument that fails a later check, such as an unsupported distance, returns 1.

## Compile

```sh
bloq compile memory.blog -d 3 --backend ir-text -o memory.bloqir
bloq compile --gallery y_memory -d 3,5 --backend ir-binary -o memory.bloq
bloq compile --gallery cnot -d 3 --fill 0 -o cnot-memory.stim
```

| Option | Meaning |
| --- | --- |
| `-d, --distance` | Required code distance; repeat or comma-separate values. Each must be odd and in `3..=255` |
| `--backend` | `stim` (default), `ir-text`, or `ir-binary` |
| `--auto-fill` | Close an open Clifford source into every compatible memory experiment and compile all variants |
| `--fill N` | Compile only compatible fill N, indexed from zero |
| `-j, --jobs N` | Concurrent distance/fill targets; positive integer. Default: available cores capped at 4 |
| `--limit FIELD=COUNT` | Override a compiler resource field; repeat for several fields, or use `unlimited` |
| `--prepare-t-with-mpps` | Use product-state resets and one stabilizer MPP instead of T-state cultivation |
| `--align-moments` | Align compatible Clifford node moments by z layer in noiseless Stim; requires `--backend stim` |
| `--clifford-proxy` | Replace T resources with perfect ports and sample a static selective path |
| `--proxy-seed SEED` | Reproducible seed, requires `--clifford-proxy` |
| `-o, --output PATH` | Output path, or base path for multiple targets |
| `-p, --print` | Write artifact data to stdout; conflicts with `--output` |

### Output files and streams

Without `--output`, files are named `<stem>-d<D>[-fill<N>].<extension>` next to the source.
Gallery runs write to the current directory.
Extensions are `.stim`, `.bloqir`, and `.bloq`.

An explicit output path is used unchanged for one target, regardless of backend.
With `-d 3,5 -o memory.bloq`, the files are `memory-d3.bloq` and
`memory-d5.bloq`. A `-fill<N>` suffix is added when several fill variants are
emitted. Choose an extension matching the backend so later commands select the
correct codec.

Status and stage progress go to stderr.
`--print` reserves stdout for artifact data, including raw binary bytes for `ir-binary`.
Stim can print several artifacts consecutively, with their labels on stderr;
stdout alone does not preserve those file boundaries. Print one target per call
when a consumer expects one circuit. The IR backends reject multi-artifact
`--print` runs.

File writes replace regular destinations atomically. Output paths that alias the
input, including hard links, are rejected, as are symlink destinations. Parent
directories must already exist. A run producing several files is not one atomic
transaction: completed files can remain if another target fails.

### Source and backend semantics

Compilation preserves the authored source, including ideal open seams.
Ports are never filled implicitly.
`--auto-fill` and `--fill` are separate Clifford experiment transforms and materialize a flat graph.
They do not accept T/selective graphs. `--fill N` selects from the same variants
as `--auto-fill`; if both are given, the selected fill wins.

The static Stim backend cannot emit `RepeatUntilSuccess` regions or unpinned guarded quantum membership.
Choose an IR backend to retain the program's control semantics.
`--clifford-proxy` is for static distance checks. It does not simulate the original T computation and does not project structural branches.
The seed is reused across that run's distances/fills and is recorded in the resulting IR metadata.

`--jobs` changes target concurrency, not emitted artifacts.
A one-target run ignores it; module compilation has its own worker policy.
Compilation stages describe current work, not a completion percentage or ETA.

### Resource limits

```sh
bloq compile input.blog -d 3 --backend ir-binary -o output.bloq \
  --limit max_witness_nodes=2000000 --limit max_boolean_steps=unlimited
```

The last override for a field wins.
Source size, allocation, realization, proof, and search caps remain finite by default; cumulative compiler Boolean work has an unlimited default.
Overrides are not total-memory or wall-clock limits.
An exhausted budget is a resource refusal, not proof of an invalid source or an unreachable choice.

## Emit saved IR

```sh
bloq emit memory.bloq -o memory.stim
bloq emit memory.bloq --backend ir-text -o memory.bloqir
bloq emit memory.bloqir --backend ir-binary -o memory.bloq
bloq emit memory.bloqir --validate --print
```

| Option | Meaning |
| --- | --- |
| `--backend` | `stim` (default), `ir-text`, or `ir-binary` |
| `--validate` | Run the full IR well-formedness audit before emission or conversion |
| `--align-moments` | Noiseless aligned Stim; rejects IR backends |
| `-o, --output PATH` | Output path; default replaces the input extension |
| `-p, --print` | Artifact data on stdout; conflicts with `--output` |

The extension selects the input codec: `.bloq` is binary and `.bloqir` is text.
For other extensions, the loader tries text and then binary. Giving binary data a
`.bloqir` extension therefore fails instead of selecting a different codec.

IR conversion preserves the program without recompiling BLOG. Decoding checks
the format, while `--validate` additionally audits the complete IR. Ordinary
emission retains its backend checks but does not run that audit. Request it for
external or hand-edited programs. When converting to the input's existing codec,
provide a different `--output` path or use `--print`: overwriting the input is
rejected.

## Inspect IR structure

```sh
bloq stats program.bloqir
bloq stats program.bloq
```

Prints a tree with compiled templates, nodes, edges, and `Static: Yes/No`.
Nodes and edges include a breakdown by concrete type across all nested graph
levels. Every stored region body counts once, independently of runtime retries.
Zero-count types are shown too.

Static means the IR contains no conditional or retry region, activation
predicate, quantum membership guard, guarded quantum seam, or discard.
Ordinary classical computation and fixed-tape conditional Pauli corrections
retain a fixed execution structure. Static does not guarantee Stim emittability.

The command reads either IR codec without flattening, recompiling, or running
the whole-program audit. Overflowing structural counts return errors. Use `bloq validate`
for an explicit audit.
Rust exposes `program.stats()?`; Python exposes `program.stats()` with read-only
fields and the same printable summary.

## Validate saved IR

```sh
bloq validate memory.bloqir
bloq validate memory.bloq
```

Runs the full IR audit and reports the first violation.
The audit checks graph structure, dependencies, boundaries, and control regions.
Success establishes IR well-formedness; it does not compare the program with an
intended logical map, simulate a shot, or guarantee that Stim can emit it.

This command accepts IR, not BLOG. Source validation occurs through graph APIs
and compilation. Use `emit --validate` when an audit should gate conversion or
Stim emission in the same invocation.

## View a graph

```sh
bloq view memory.blog
bloq view --gallery cnot --html
bloq view --gallery cnot --gltf --pop-face=-Y
bloq view --gallery three_bit_adder --module-view --html
bloq view cnot.bloqir --svg --include-classical -o cnot-ir.svg
```

| Option | Meaning |
| --- | --- |
| `--gltf` | Generate a graph model |
| `--html` | Generate an HTML viewer with its model embedded |
| `--svg` | Render saved `.bloqir` or `.bloq` IR as a nested dependency graph |
| `--include-classical` | Show classical nodes and their edges in IR SVG |
| `--module-view` | Color source geometry by module definition, with a legend; the hierarchy is preserved |
| `-o, --output PATH` | IR SVG path; default replaces the input extension with `.svg` |
| `--pipe-length FLOAT` | Rendered pipe length, default `2.0` |
| `--pop-face DIRECTION` | Remove outward faces toward `+X`, `-X`, `+Y`, `-Y`, `+Z`, or `-Z`; repeat for several directions |

Without a view flag, BLOG generates both model and HTML; saved IR generates SVG.
It writes beside the source or in the current directory for gallery entries.
For BLOG, `--gltf` and `--html` may be combined; `--output` and
`--include-classical` apply only to saved IR. Geometry exports use a flattened
rendering projection by default. `--module-view` retains the hierarchy and colors
it by definition without changing the source. The model shows logical geometry,
not a physical execution schedule.

The IR SVG preserves region nesting and typed dependencies; it does not unroll
retries or instantiate physical gates. Classical nodes are hidden unless
`--include-classical` is given. SVG generation does not run a full IR audit.

## Gallery and completions

```sh
bloq gallery
bloq completion zsh > _bloq
```

The gallery is versioned with the installed software; use its listing instead of assuming a historical list is current.
Completion shells are `bash`, `elvish`, `fish`, `powershell`, and `zsh`.
Install the generated script according to your shell's completion setup.
