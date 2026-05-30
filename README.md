# Overdrive 🏎️

Overdrive is a performance-focused Terminal User Interface (TUI) and CLI tool designed specifically to orchestrate, monitor, and manage concurrent [vLLM](https://github.com/vllm-project/vllm) execution instances locally on your **NVIDIA DGX Spark**.

Built entirely using `Python` and `Textual`, Overdrive bridges the gap between raw hardware capabilities and local multi-agent or engineering workflows, bypassing brittle, manually typed setup commands.

## ✨ Features

- **Automated Storage Scanning:** Dynamically parses local Hugging Face storage paths and TrueNAS ZFS dataset mounts to list local models and map architectural metadata automatically.
- **Concurrent Runtimes:** Spin up, isolate, and maintain multiple vLLM instances in parallel with built-in port collision protection.
- **NVIDIA NGC Stack Integration:** Native management utilising optimised `nvcr.io/nvidia/vllm:26.04-py3` container configurations via the official Docker SDK.
- **Asynchronous Live Logs:** Intercepts and streams real-time console telemetry from processing models cleanly inside the dashboard layout without UI locking.

## 🚀 Getting Started

### Prerequisites

Ensure the following environments are active on your host system:
- **NVIDIA GPU Drivers & Container Toolkit**
- **Docker Engine**
- **Python 3.10+**

### Installation

```bash
# Clone the repository
git clone [https://github.com/yourusername/overdrive.git](https://github.com/yourusername/overdrive.git)
cd overdrive

# Set up virtual environment and install packages
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
