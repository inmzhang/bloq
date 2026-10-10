# Installation

## Python

Requires Python 3.10 or newer. The PyPI distribution is `bloq-py`; the Python
import and command are `bloq`. These commands install the first public release.
Do not install the unrelated PyPI distribution `bloq` in the same environment.

::::{md-tab-set}
:::{md-tab-item} uv
```sh
uv add "bloq-py==0.1.1"
```
:::
:::{md-tab-item} pip
```sh
pip install "bloq-py==0.1.1"
```
:::
::::

### Platform Support

CI builds prebuilt wheels for:

| Platform | Architecture |
| --- | --- |
| Linux (manylinux) | x86_64 |
| macOS | ARM64 (Apple Silicon) |
| Windows | x86_64 |

Otherwise, installation builds from source and requires Rust 1.95.0 or newer.

## Rust

Requires Rust 1.95.0 or newer. Add Bloq to your Cargo project:

```sh
cargo add bloq@=0.1.1
```

See [Cargo feature flags](../guide.md#cargo-feature-flags) for optional features.

## CLI

::::{md-tab-set}
:::{md-tab-item} uv
```sh
uv tool install "bloq-py==0.1.1"
```
:::
:::{md-tab-item} cargo-binstall
```sh
cargo binstall bloq-cli --version 0.1.1
```
:::
:::{md-tab-item} Cargo
```sh
cargo install bloq-cli --version 0.1.1 --locked
```
:::
::::

The executable is `bloq`. The Python package also includes the CLI.

## Editor Application

### Browser editor

The WebAssembly target is `wasm32-unknown-unknown`.
Open [bloqec.com/editor/](https://bloqec.com/editor/).

:::{important}
The browser editor requires **WebGPU** and a compatible GPU/driver.

- **Chrome:** WebGPU is [enabled by default on supported devices](https://developer.chrome.com/docs/web-platform/webgpu/troubleshooting-tips). Keep graphics acceleration enabled.
- **Firefox:** If WebGPU is disabled, open `about:config`, search for [`dom.webgpu.enabled`](https://developer.mozilla.org/en-US/docs/Mozilla/Firefox/Experimental_features#webgpu_api), set it to `true`, and restart Firefox.
:::

### Desktop app

::::{md-tab-set}
:::{md-tab-item} cargo-binstall
```sh
cargo binstall bloq_editor --version 0.1.1
```
:::
:::{md-tab-item} Cargo
```sh
cargo install bloq_editor --version 0.1.1 --locked
```
:::
::::

Run `bloq_editor` to launch the application.

#### Prebuilt binaries

The [GitHub release workflow](https://github.com/inmzhang/bloq/blob/main/.github/workflows/binaries.yml)
publishes these editor archives. Download from [GitHub Releases](https://github.com/inmzhang/bloq/releases)
or use curl:

| Platform | Download |
| --- | --- |
| Linux x86_64 | [tar.gz](https://github.com/inmzhang/bloq/releases/download/v0.1.1/bloq_editor-x86_64-unknown-linux-gnu.tar.gz) |
| macOS ARM64 | [tar.gz](https://github.com/inmzhang/bloq/releases/download/v0.1.1/bloq_editor-aarch64-apple-darwin.tar.gz) |
| Windows x86_64 | [zip](https://github.com/inmzhang/bloq/releases/download/v0.1.1/bloq_editor-x86_64-pc-windows-msvc.zip) |

::::{md-tab-set}
:::{md-tab-item} Linux
```sh
curl -fL https://github.com/inmzhang/bloq/releases/download/v0.1.1/bloq_editor-x86_64-unknown-linux-gnu.tar.gz -o bloq-editor.tar.gz
mkdir -p bloq-editor
tar -xzf bloq-editor.tar.gz -C bloq-editor
./bloq-editor/bloq_editor
```
:::
:::{md-tab-item} macOS
```sh
curl -fL https://github.com/inmzhang/bloq/releases/download/v0.1.1/bloq_editor-aarch64-apple-darwin.tar.gz -o bloq-editor.tar.gz
mkdir -p bloq-editor
tar -xzf bloq-editor.tar.gz -C bloq-editor
./bloq-editor/bloq_editor
```
:::
:::{md-tab-item} Windows
```powershell
curl.exe -fL https://github.com/inmzhang/bloq/releases/download/v0.1.1/bloq_editor-x86_64-pc-windows-msvc.zip -o bloq-editor.zip
Expand-Archive -Path bloq-editor.zip -DestinationPath bloq-editor
.\bloq-editor\bloq_editor.exe
```
:::
::::

## Source and development setup

### Prerequisites

Depending on your platform, source builds may require a C/C++ toolchain and
system libraries, including display/audio libraries for the desktop editor.
The development workspace requires Rust 1.98.0 or newer.

### Checkout and development tools

Required development tools:

| Tool | Purpose |
| --- | --- |
| Git | Repository checkout |
| [rustup](https://rustup.rs/) | Pinned Rust toolchain, Cargo, rustfmt, and Clippy |
| Python 3.10+ | Python bindings and tests; 3.11+ for documentation tooling |
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | Python environments and dependencies |
| [just](https://just.systems/man/en/) | Workspace command recipes |
| [cargo-nextest](https://github.com/nextest-rs/nextest) | Rust test runner |

Web editor development additionally requires the Bevy CLI; its installation
command appears below.

```sh
git clone https://github.com/inmzhang/bloq.git
cd bloq
rustup toolchain install
cargo install just cargo-nextest --locked
```

Run the following commands from the repository root. Rustup selects the pinned
toolchain and its `rustfmt` and `clippy` components from `rust-toolchain.toml`.

### Python development

Create the editable installation and install the locked development dependencies:

```sh
just py-develop
uv run --project bloq_py --no-sync python -c "import bloq; print(bloq.__version__)"
uv run --project bloq_py --no-sync bloq --help
```

The environment lives in `bloq_py/.venv`. To build distributable wheels or a
source archive:

```sh
just py-build
just py-sdist
```

Build outputs are written to `target/wheels/`.

### Rust library and CLI

This checkout's Rust APIs use glam 0.33 geometry types. Use a matching glam
version when constructing vectors for those APIs.

```sh
cargo build -p bloq --locked
cargo run -p bloq-cli --locked -- --help
```

To install the CLI from this checkout:

```sh
cargo install --path bloq_cli --locked
```

### Desktop editor

After installing the system dependencies above, run the editor:

```sh
just editor
```

Or install it from the checkout:

```sh
cargo install --path bloq_editor --locked
bloq_editor
```

### Web editor development

Install the WebAssembly target and the Bevy CLI used by the website build:

```sh
rustup target add wasm32-unknown-unknown
cargo install --git https://github.com/TheBevyFlock/bevy_cli \
  --tag cli-v0.1.0-alpha.2 --locked bevy_cli
just web
```

This builds and opens the editor locally. Use a WebGPU-enabled browser as
described above.

### Verification

```sh
just fmt-check
just test bloq           # clippy, Rust tests, and doctests
just py-test            # Python bindings
just ci                 # complete CI gate
```

After changing Python bindings, regenerate the stubs with `just py-stub`.
Use `just --list` to see all workspace recipes and the
[contributor guide](../development/contributing.md) for contribution policy.
