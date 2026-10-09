# Website Maintenance

The website uses one documentation tree, `docs/`. Sphinx renders Markdown and
the Python API; `cargo doc` supplies the Rust API. The browser editor is built
separately and assembled with those outputs.

## Build and preview

Use the [development tools](getting-started/installation.md#checkout-and-development-tools)
and WebAssembly setup from the installation page. The Bevy CLI version and
build environment are pinned in `.github/workflows/site-build.yml`.

```sh
just site-docs       # fast documentation preview
just site-preview   # serve target/site on port 8000
```

| Local path | Content |
| --- | --- |
| `/` | Homepage in the development documentation version |
| `/docs/dev/` | Development documentation |
| `/editor/` | Current browser editor |

Before publishing, assemble and check the complete site:

```sh
just site
just site-check
```

A local editor bundle can be reused when it matches the checked-out source:

```sh
just site --editor-bundle target/bevy_web/web-release/bloq_editor
```

Output stays under `target/site/`. The checker verifies local links, anchors,
assets, version metadata, and accidental private/build-host paths. Reused WASM
must carry the same host-path remapping as the normal editor build.

## Maintain the chapters

Edit chapter sources in `docs/`. Update `docs/index.md` for site navigation and
`docs/user-guide.md` for the User Guide's chapter order.

- Explain current behavior with short prose, tables, and diagrams.
- Keep runnable examples in `docs/examples/` and include them with `literalinclude`.
- After API or IR changes, rerun affected examples in temporary directories.
  Validate saved IR and refresh its downloads, captured output, and diagrams together.

Use these directives for illustrations:

| Directive or option | Use |
| --- | --- |
| `bloq-view` | Interactive block graph from a gallery entry, or a BLOG file with `:source:` |
| `:modules:` | Show module ownership |
| `:surface:` | Show a correlation surface from a Pauli word over lexicographically sorted Ports |
| `:measurement:` | Show a named measurement's correlation surface |
| `:pop-faces:` | Expose the surface through selected face directions |
| `detector-slices` | Embed a generated physical-construction diagram |

Correlation overlays show support. Named-measurement illustrations project
structural branches to their true arms.
See [Update visual assets](#update-visual-assets) for regeneration tools.

## Generate the API references

Python signatures come from generated stubs and docstrings from the bindings.
The functional reference in `docs/api/python.rst` lists each public export once;
`bloq_py/tests/test_docs.py` checks that coverage.

```sh
just py-stub        # after changing binding annotations
just py-docs        # rebuild canonical Sphinx sources
just doc-check      # check public Rust documentation
```

Old Python per-object URLs redirect to inline API anchors, preserving method
fragments. Do not hand-edit generated stubs or duplicate binding docstrings.

## Retain release versions

| URL | Source |
| --- | --- |
| `/docs/dev/` | Working development source |
| `/docs/TAG/` | An existing, immutable release tag |
| `/docs/stable/` | A selected retained release snapshot |

```sh
just site --tag TAG
just site --stable TAG
```

Replace `TAG` with an existing release tag. Tagged prose, Python bindings, and
Rust API are built from that same checkout. An existing snapshot cannot be
replaced by a different commit. `versions.json` records each version's source
and pages; the selector retains the current page when it exists in the target.

The `gh-pages` branch retains release directories. Preserve those directories
when assembling a new version. The hosted editor shows its own current version;
historical documentation does not select a historical editor build.

## Publish a reviewed build

CI produces a validated `bloq-site-preview` artifact for website changes.
Review that artifact before publication. Run the website workflow with `publish`
enabled to publish; optional `release_tag` adds a release snapshot and
`make_stable` selects it. Package releases are separate; see [Releases](releasing.md).

```{important}
Review the assembled site before changing the deployed version. Keep previous
release snapshots when publishing.
```

The workflow restores retained snapshots, builds the new tree, and deploys that
same tree to GitHub Pages. Domain and HTTPS settings belong to repository
maintainers; the generated `CNAME` names `bloqec.com`.

## Update visual assets

Site styling lives in `docs/_static/site.css`; logo sources live in
`docs/assets/logos/`. Use `bloq-view` for interactive graphs and the public IR
export API for dependency diagrams.

| Asset | Generator |
| --- | --- |
| Blocks and Pipes thumbnails | `tools/render_block_thumbnails.py` |
| Gallery pages and thumbnails | `tools/render_gallery_examples.py` |
| 2D companions to paper figures | `tools/render_viewer_companions.py --paper-figures <directory>` |
| Retained tutorial plots | `tools/plot_tutorial_results.py` |
| Physical construction diagrams | `tools/render_construction_slices.py` |

Run the thumbnail and gallery generators with
`uv run --project bloq_py --no-sync python <script>`.
Before generating gallery thumbnails, build the editor with
`cargo build -p bloq_editor --locked`. Use the generator's `--editor` option
to select another binary.

Keep experiment settings beside their results, and record source URLs and
hashes in `docs/assets/source-provenance.json`. Replot retained statistical data
without resampling. Historical circuit bundles keep their original circuits
and metadata for replay.

To regenerate the THTH occupation figure, use the settings in its saved caption.
Update its SVG, interval CSV, caption, and provenance hashes together.
