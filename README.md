# Overdrive 🏎️

Overdrive is a performance-focused Terminal User Interface (TUI) and CLI tool designed specifically to orchestrate, monitor, and manage concurrent [vLLM](https://github.com/vllm-project/vllm) execution instances locally on your **NVIDIA DGX Spark**.

Built entirely using `Python` and `Textual`, Overdrive bridges the gap between raw hardware capabilities and local multi-agent or engineering workflows, bypassing brittle, manually typed setup commands.

## ✨ Features

- **Automated Model Discovery:** Scans a Hugging Face cache root for model snapshots, extracts config metadata, and applies profile overrides automatically.
- **Concurrent Runtimes:** Spin up, isolate, and maintain multiple vLLM instances in parallel with built-in port collision protection.
- **Preflight Admission Control:** Estimate launch memory usage, account for active managed reservations, and reject launches that exceed a configured GPU budget.
- **Live Operations Dashboard:** Monitor managed containers, tail logs, and inspect live Docker stats from the CLI or the Textual TUI.
- **Hugging Face CLI Integration:** Search Hub models and download them through the real `hf` CLI without leaving Overdrive.
- **NVIDIA NGC Stack Integration:** Manage `nvcr.io/nvidia/vllm:26.04-py3` container configurations through the official Docker SDK.

## 🚀 Getting Started

### Prerequisites

Ensure the following environments are active on your host system:
- **NVIDIA GPU Drivers & Container Toolkit**
- **Docker Engine**
- **Python 3.10+**

### Installation

```bash
# Clone the repository
git clone https://github.com/jordanovski/overdrive.git
cd overdrive

# Set up virtual environment and install packages
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
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
overdrive tui
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
