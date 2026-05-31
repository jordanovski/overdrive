# Overdrive 🏎️

Overdrive is a performance-focused web UI and CLI tool designed specifically to orchestrate, monitor, manage, and benchmark concurrent [vLLM](https://github.com/vllm-project/vllm) execution instances locally on your **NVIDIA DGX Spark**.

Built entirely using `Python` and `FastAPI`, Overdrive bridges the gap between raw hardware capabilities and local multi-agent or engineering workflows, bypassing brittle, manually typed setup commands.

## ✨ Features

- **Automated Model Discovery:** Scans a Hugging Face cache root for model snapshots, extracts config metadata, and applies profile overrides automatically.
- **Concurrent Runtimes:** Spin up, isolate, and maintain multiple vLLM instances in parallel with built-in port collision protection.
- **Preflight Admission Control:** Estimate launch memory usage, account for active managed reservations, and reject launches that exceed a configured GPU budget.
- **Live Operations Dashboard:** Monitor managed containers, tail logs, inspect live Docker stats, and control launches from the browser-based web console or the CLI.
- **SWE-bench Benchmark Page:** Select multiple local models, run SWE-bench sequentially with recommended vLLM settings, and compare resolution rates in a built-in results graph.
- **Hugging Face CLI Integration:** Search Hub models and download them through the real `hf` CLI without leaving Overdrive.
- **NVIDIA NGC Stack Integration:** Manage `nvcr.io/nvidia/vllm:26.04-py3` container configurations through the official Docker SDK.

## 🚀 Getting Started

### Prerequisites

Ensure the following environments are active on your host system:

- **NVIDIA GPU Drivers & Container Toolkit**
- **Docker Engine**
- **Python 3.10+**

### Preferred On DGX: Docker Compose + GHCR Image

You do **not** need to run `overdrive web ...` yourself when using the container. The
image already starts the web server on container boot.

You also do **not** need to clone this repository onto the DGX. The GitHub Action now
builds and publishes the container to GitHub Container Registry at:

```text
ghcr.io/jordanovski/overdrive:latest
```

Fast path:

```bash
mkdir -p ~/overdrive
cd ~/overdrive
curl -O https://raw.githubusercontent.com/jordanovski/overdrive/main/compose.yaml
curl -O https://raw.githubusercontent.com/jordanovski/overdrive/main/.env.example
cp .env.example .env
# edit .env and set OVERDRIVE_HUB_ROOT to the host path that contains your models
docker compose up -d
```

Then open `http://localhost:8080` in your browser.

If the repository or package is private, log in once on the DGX before the first pull:

```bash
echo "$GHCR_TOKEN" | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
```

That token needs `read:packages` access.

Important:

- The model directory must be mounted into the Overdrive container at the same absolute host path.
- That model directory mount must be writable if you want the Model Search page to download repos into OVERDRIVE_HUB_ROOT.
- Overdrive uses the host Docker socket to launch managed vLLM containers.
- The Compose file also maps `host.docker.internal` to the host gateway so benchmark jobs can reach host-published vLLM ports from inside the Overdrive container.
- Benchmark artifacts and saved profiles persist in Docker volumes managed by Compose.

To pull the latest published container later:

```bash
docker compose pull
docker compose up -d
```

Stop the stack with:

```bash
docker compose down
```

### Install Without a Source Checkout

For normal usage on a DGX or workstation, install Overdrive as a packaged CLI so you can
run `overdrive` from anywhere without keeping the repository checked out locally.

#### Preferred: `pipx`

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y pipx python3-venv
pipx ensurepath
exec "$SHELL" -l

# Install from a published wheel or directly from Git
pipx install /path/to/overdrive-0.1.0-py3-none-any.whl
# or
pipx install "git+https://github.com/jordanovski/overdrive.git"
```

Verify the install:

```bash
which overdrive
overdrive --help
```

### Run The Web Console

Once installed, start the web UI locally:

```bash
overdrive --hub-root /raid/huggingface web --host 0.0.0.0 --port 8080
```

Then open `http://localhost:8080` in your browser.

The web console now includes two pages:

- `/` for launch control, runtime status, logs, and profile management
- `/benchmarks` for SWE-bench runs across multiple selected local models

### SWE-bench Notes

The benchmark page launches each selected model one by one with Overdrive's recommended
settings, generates predictions against the chosen SWE-bench dataset, runs the official
`swebench.harness.run_evaluation` harness, and plots the resulting resolution rate.

Practical constraints:

- Start with `princeton-nlp/SWE-bench_Lite` and a small instance limit before trying larger runs.
- SWE-bench evaluation is Docker-heavy and can consume significant CPU, disk, and time.
- On ARM64 hosts, Overdrive asks SWE-bench to build evaluation images locally instead of pulling the default x86 namespace.

### Run As A Docker Container

If you prefer not to use Compose, the direct container form is still available.
You still do **not** run `overdrive web` manually inside the container; the image's
default command starts the web UI for you.

Pull the published image directly from GitHub Container Registry:

```bash
docker pull ghcr.io/jordanovski/overdrive:latest
```

Run it against the host Docker daemon and your model root:

```bash
docker run --rm -p 8080:8080 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /raid/huggingface:/raid/huggingface \
  -e OVERDRIVE_HUB_ROOT=/raid/huggingface \
  ghcr.io/jordanovski/overdrive:latest
```

Important: mount the model directory into the Overdrive container at the same absolute
path it has on the host. Keep that mount writable if you want Overdrive to download
models into it from the Model Search page. Overdrive passes discovered model paths
through to Docker when it launches vLLM containers, so path parity matters.

For benchmark runs from inside the Overdrive container, also make sure the container has
enough disk available for SWE-bench build and evaluation artifacts.

#### Update an existing install

```bash
# Reinstall from a newer wheel
pipx install --force /path/to/overdrive-0.1.0-py3-none-any.whl

# or refresh from Git
pipx install --force "git+https://github.com/jordanovski/overdrive.git"
```

#### Remove Overdrive

```bash
pipx uninstall overdrive
```

#### No `sudo` available

If you cannot install `pipx` from the OS package manager, create one dedicated virtual
environment for Overdrive and expose its console script on your `PATH`:

```bash
python3 -m venv ~/.local/share/overdrive
~/.local/share/overdrive/bin/pip install --upgrade pip
~/.local/share/overdrive/bin/pip install /path/to/overdrive-0.1.0-py3-none-any.whl

mkdir -p ~/.local/bin
ln -sf ~/.local/share/overdrive/bin/overdrive ~/.local/bin/overdrive
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
exec "$SHELL" -l
```

To update this style of install, rerun the `pip install` command against a newer wheel.
To remove it, delete the environment and launcher:

```bash
rm -rf ~/.local/share/overdrive
rm -f ~/.local/bin/overdrive
```

### Development Setup

If you are contributing to Overdrive itself, use a source checkout instead:

```bash
git clone https://github.com/jordanovski/overdrive.git
cd overdrive
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

### Project Layout

```text
scr/   Python source code
test/  Automated tests
```

### Development

```bash
pytest
ruff check .
```

### CLI Commands

```bash
overdrive --hub-root ~/.cache/huggingface/hub scan
overdrive --profiles ~/.config/overdrive/profiles.yaml scan
overdrive scan
overdrive scan --json-output
overdrive plan org/model-7b --gpu-memory-budget-gb 80
overdrive plan org/model-7b --json-output
overdrive profile org/model-7b --preferred-port 8002 --max-model-len 32768
overdrive profile org/model-7b --json-output --preferred-port 8002
overdrive up org/model-7b --dry-run
overdrive up org/model-7b --json-output --dry-run
overdrive stop org/model-7b --json-output
overdrive cleanup --json-output
overdrive ps
overdrive logs overdrive-org-model-7b --tail 100
overdrive stats
overdrive stats --watch --interval 2 --samples 5
overdrive stats --json-output
overdrive stats --json-output --watch --interval 2 --samples 5
overdrive models-search qwen --limit 5
overdrive models-search llama --author meta-llama --json-output
overdrive models-download Qwen/Qwen3-8B --dry-run
overdrive models-download Qwen/Qwen3-8B --local-dir .\models\qwen3-8b
overdrive web --host 0.0.0.0 --port 8080
```

Use `hf auth login` first if you need access to gated or private repositories. Overdrive's
Hugging Face integration shells out to the real `hf` CLI for model discovery and downloads.

The top-level `--hub-root` option overrides the Hugging Face cache root used by `scan`, and
`--profiles` overrides the YAML profile path.

### Profiles

Overdrive stores model-specific launch overrides in `~/.config/overdrive/profiles.yaml`.
These profiles can define preferred ports, max model length, tensor parallel size,
GPU memory budgets, and additional vLLM launch arguments.

Use `overdrive plan <model>` to run a preflight check before launch. It reports the
projected port assignment, estimated model footprint, active managed reservations,
and whether the total reservation fits within the configured GPU memory budget.
