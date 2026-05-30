"""Textual application for Overdrive orchestration."""

from __future__ import annotations

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    RichLog,
    Static,
)

from overdrive.hardware import (
    GPUDevice,
    detect_gpus,
    recommended_gpu_budget_gb,
    recommended_kv_cache_dtype,
    recommended_max_model_len,
    recommended_tensor_parallel_size,
)
from overdrive.models import ModelMetadata, ModelProfile, PreflightReport
from overdrive.state import EngineStateManager


def _recommended_port(manager: EngineStateManager, model: ModelMetadata) -> int:
    return manager.runtime.reserve_port(model.profile.preferred_port)


def _display_dtype(model: ModelMetadata) -> str:
    if model.dtype != "unknown":
        return model.dtype
    text_config = model.config_data.get("text_config", {})
    if isinstance(text_config, dict):
        nested_dtype = text_config.get("dtype")
        if isinstance(nested_dtype, str) and nested_dtype:
            return nested_dtype
    return "unknown"


class ExitConfirmationScreen(ModalScreen[str]):
    CSS = """
    ExitConfirmationScreen {
        align: center middle;
    }

    #dialog {
        width: 60;
        height: auto;
        border: thick $accent;
        padding: 1 2;
        background: $surface;
    }

    .dialog-button {
        margin-top: 1;
    }
    """

    def __init__(self, active_count: int) -> None:
        super().__init__()
        self.active_count = active_count

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(
                (
                    f"{self.active_count} managed container(s) are still running.\n"
                    "Clean them up before quitting, or leave them alive as daemons?"
                )
            )
            yield Button("Cleanup And Quit", id="cleanup", variant="error", classes="dialog-button")
            yield Button("Keep Alive And Quit", id="keep-alive", classes="dialog-button")
            yield Button("Cancel", id="cancel", classes="dialog-button")

    @on(Button.Pressed)
    def handle_choice(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id)


class OverdriveApp(App[None]):
    CSS = """
    Screen {
        layout: vertical;
    }

    #main {
        height: 1fr;
    }

    #models-pane {
        width: 34%;
        border: solid $accent;
    }

    #details-pane {
        width: 66%;
        border: solid $primary;
        padding: 1;
    }

    #model-list {
        height: 1fr;
    }

    .field {
        margin-bottom: 1;
    }

    .field-label {
        margin-top: 1;
        margin-bottom: 0;
    }

    #actions {
        height: auto;
        margin-top: 1;
    }

    #actions Button {
        width: 1fr;
        margin-right: 1;
    }

    #actions Button:last-of-type {
        margin-right: 0;
    }

    #log {
        height: 1fr;
        border: round $secondary;
    }
    """

    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("p", "plan_selected", "Plan Selected"),
        ("l", "launch_selected", "Launch Selected"),
        ("w", "save_profile", "Save Profile"),
        ("s", "stop_selected", "Stop Selected"),
        ("c", "cleanup_managed", "Cleanup Managed"),
        ("ctrl+c", "quit", "Quit"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, manager: EngineStateManager) -> None:
        super().__init__()
        self.manager = manager
        self.models: list[ModelMetadata] = []
        self.selected: ModelMetadata | None = None
        self.gpus: list[GPUDevice] = detect_gpus()

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with Vertical(id="models-pane"):
                yield Static("Discovered Models")
                yield ListView(id="model-list")
            with Vertical(id="details-pane"):
                yield Static("Select a model to edit launch settings.", id="summary")
                yield Static("No active Overdrive containers.", id="active-summary")
                yield Static("No live stats available.", id="stats-summary")
                yield Static("Port", classes="field-label")
                yield Input(placeholder="Port", id="port", classes="field")
                yield Static("Max Model Length", classes="field-label")
                yield Input(placeholder="Max model length", id="max-model-len", classes="field")
                yield Static("Tensor Parallel Size (GPUs)", classes="field-label")
                yield Input(
                    placeholder="Tensor parallel size (GPUs)",
                    value="1",
                    id="tensor-parallel",
                    classes="field",
                )
                yield Static("KV Cache DType", classes="field-label")
                yield Input(
                    placeholder="KV cache dtype (auto|fp8|fp8_e4m3|fp8_e5m2)",
                    id="kv-cache-dtype",
                    classes="field",
                )
                yield Static("GPU Memory Budget (GiB)", classes="field-label")
                yield Input(placeholder="GPU memory budget (GiB)", id="gpu-budget", classes="field")
                with Horizontal(id="actions"):
                    yield Button("Plan Launch", id="plan", variant="primary")
                    yield Button("Launch Model", id="launch", variant="success")
                    yield Button("Save Profile", id="save-profile")
                    yield Button("Stop Model", id="stop", variant="warning")
                    yield Button("Cleanup Managed", id="cleanup", variant="error")
                yield RichLog(id="log")
        yield Footer()

    def on_mount(self) -> None:
        self.action_refresh()
        self.set_interval(2.0, self.refresh_dashboard)

    def action_refresh(self) -> None:
        self.models = self.manager.discover_models()
        listing = self.query_one("#model-list", ListView)
        listing.clear()
        for model in self.models:
            listing.append(ListItem(Label(model.display_name), name=model.model_id))
        self._log(f"Loaded {len(self.models)} model(s) from {self.manager.hub_root}")

    def _populate_fields(self, model: ModelMetadata) -> None:
        port = _recommended_port(self.manager, model)
        max_model_len = recommended_max_model_len(model, self.gpus)
        tensor_parallel = recommended_tensor_parallel_size(model, self.gpus)
        kv_cache_dtype = recommended_kv_cache_dtype(model, self.gpus)
        gpu_budget = recommended_gpu_budget_gb(model, self.gpus)
        hardware_summary = (
            "No GPU telemetry available"
            if not self.gpus
            else "; ".join(
                f"{gpu.name}: free {gpu.free_memory_gb:g}/{gpu.total_memory_gb:g} GiB"
                for gpu in self.gpus
            )
        )
        self.query_one("#summary", Static).update(
            (
                f"{model.model_id}\n"
                f"Architecture: {model.architecture}\n"
                f"DType: {_display_dtype(model)}\n"
                f"Hardware: {hardware_summary}\n"
                f"Recommended launch: port {port}, max len {max_model_len}, "
                f"tensor parallel {tensor_parallel}, kv cache {kv_cache_dtype}, "
                f"budget {gpu_budget:g} GiB"
            )
        )
        self.query_one("#port", Input).value = str(port)
        self.query_one("#max-model-len", Input).value = str(max_model_len)
        self.query_one("#tensor-parallel", Input).value = str(tensor_parallel)
        self.query_one("#kv-cache-dtype", Input).value = kv_cache_dtype
        self.query_one("#gpu-budget", Input).value = str(gpu_budget)

    def _current_int(self, widget_id: str) -> int | None:
        value = self.query_one(widget_id, Input).value.strip()
        return int(value) if value else None

    def _current_float(self, widget_id: str) -> float | None:
        value = self.query_one(widget_id, Input).value.strip()
        return float(value) if value else None

    def _current_kv_cache_dtype(self) -> str | None:
        return self.query_one("#kv-cache-dtype", Input).value.strip() or None

    def _current_launch_settings(self) -> dict[str, int | float | str | None]:
        return {
            "preferred_port": self._current_int("#port"),
            "max_model_len": self._current_int("#max-model-len"),
            "tensor_parallel_size": self._current_int("#tensor-parallel") or 1,
            "kv_cache_dtype": self._current_kv_cache_dtype(),
            "gpu_memory_budget_gb": self._current_float("#gpu-budget"),
        }

    def _format_preflight_report(self, report: PreflightReport) -> str:
        parts = [
            f"allowed={report.allowed}",
            f"port={report.requested_port}",
            f"estimated_model_memory_gb={report.estimated_model_memory_gb}",
            f"active_reserved_memory_gb={report.active_reserved_memory_gb}",
            f"total_reserved_memory_gb={report.total_reserved_memory_gb}",
        ]
        if report.reason:
            parts.append(f"reason={report.reason}")
        return " | ".join(parts)

    def _log(self, message: str) -> None:
        self.query_one("#log", RichLog).write(message)

    def refresh_dashboard(self) -> None:
        containers = self.manager.active_containers()
        stats = {item.name: item for item in self.manager.runtime.list_managed_stats()}
        if not containers:
            self.query_one("#active-summary", Static).update("No active Overdrive containers.")
            self.query_one("#stats-summary", Static).update("No live stats available.")
            return

        lines = [
            (
                f"{container.model_id or container.name} | "
                f"{container.status} | port {container.host_port}"
            )
            for container in containers
        ]
        self.query_one("#active-summary", Static).update("\n".join(lines))

        stat_lines = []
        for container in containers:
            snapshot = stats.get(container.name)
            if snapshot is None:
                continue
            stat_lines.append(
                (
                    f"{container.name} | cpu {snapshot.cpu_percent}% | "
                    f"mem {snapshot.memory_usage_gb}/{snapshot.memory_limit_gb} GiB | "
                    f"net {snapshot.network_rx_mb}/{snapshot.network_tx_mb} MB"
                )
            )
        self.query_one("#stats-summary", Static).update(
            "\n".join(stat_lines) if stat_lines else "No live stats available."
        )

        if self.selected is None:
            return

        container = next(
            (item for item in containers if item.model_id == self.selected.model_id),
            None,
        )
        if container is None:
            return

        log = self.query_one("#log", RichLog)
        log.clear()
        for line in self.manager.runtime.stream_logs(container.name, tail=25):
            log.write(line)

    @on(ListView.Selected)
    def handle_selected(self, event: ListView.Selected) -> None:
        self.selected = next(
            (model for model in self.models if model.model_id == event.item.name),
            None,
        )
        if self.selected is not None:
            self._populate_fields(self.selected)

    @on(Button.Pressed, "#launch")
    def launch_button(self) -> None:
        self.action_launch_selected()

    @on(Button.Pressed, "#plan")
    def plan_button(self) -> None:
        self.action_plan_selected()

    @on(Button.Pressed, "#save-profile")
    def save_profile_button(self) -> None:
        self.action_save_profile()

    @on(Button.Pressed, "#stop")
    def stop_button(self) -> None:
        self.action_stop_selected()

    @on(Button.Pressed, "#cleanup")
    def cleanup_button(self) -> None:
        self.action_cleanup_managed()

    def action_launch_selected(self) -> None:
        model = self.selected
        if model is None:
            self.notify("Select a model first.")
            return

        settings = self._current_launch_settings()

        def worker() -> None:
            result = self.manager.launch_model(
                model.model_id,
                preferred_port=settings["preferred_port"],
                max_model_len=settings["max_model_len"],
                tensor_parallel_size=settings["tensor_parallel_size"],
                kv_cache_dtype=settings["kv_cache_dtype"],
                gpu_memory_budget_gb=settings["gpu_memory_budget_gb"],
            )
            self.call_from_thread(
                self._log,
                f"{result.status}: {result.container_name} on port {result.host_port}",
            )
            self.call_from_thread(self.refresh_dashboard)

        self.run_worker(worker, thread=True)

    def action_plan_selected(self) -> None:
        model = self.selected
        if model is None:
            self.notify("Select a model first.")
            return

        report = self.manager.preflight_launch(model.model_id, **self._current_launch_settings())
        self._log(self._format_preflight_report(report))

    def action_save_profile(self) -> None:
        model = self.selected
        if model is None:
            self.notify("Select a model first.")
            return

        settings = self._current_launch_settings()
        profile = ModelProfile(
            preferred_port=settings["preferred_port"],
            max_model_len=settings["max_model_len"],
            tensor_parallel_size=settings["tensor_parallel_size"] or 1,
            kv_cache_dtype=settings["kv_cache_dtype"],
            gpu_memory_utilization=model.profile.gpu_memory_utilization,
            gpu_memory_budget_gb=settings["gpu_memory_budget_gb"],
            extra_args=list(model.profile.extra_args),
        )
        path = self.manager.save_profile(model.model_id, profile)
        self._log(f"saved profile for {model.model_id} to {path}")
        self.action_refresh()
        refreshed = self.manager.get_model(model.model_id)
        self.selected = refreshed
        self._populate_fields(refreshed)

    def action_stop_selected(self) -> None:
        model = self.selected
        if model is None:
            self.notify("Select a model first.")
            return
        stopped = self.manager.stop_model(model.model_id)
        self._log(f"stop {'succeeded' if stopped else 'found nothing'} for {model.model_id}")
        self.refresh_dashboard()

    def action_cleanup_managed(self) -> None:
        count = self.manager.cleanup()
        self._log(f"cleaned up {count} managed container(s)")
        self.refresh_dashboard()

    def action_quit(self) -> None:
        active = self.manager.active_containers()
        if not active:
            self.exit()
            return

        self.push_screen(
            ExitConfirmationScreen(len(active)),
            self._handle_exit_choice,
        )

    def _handle_exit_choice(self, choice: str | None) -> None:
        if choice == "cleanup":
            count = self.manager.cleanup()
            self._log(f"cleaned up {count} managed container(s) on exit")
            self.exit()
            return
        if choice == "keep-alive":
            self._log("leaving managed containers running")
            self.exit()