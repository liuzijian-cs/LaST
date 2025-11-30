import math
import lightning as L
from utils import Color as Co
from rich.style import Style
from datetime import timedelta
from rich.console import Console, RenderableType
from rich.progress import BarColumn, Progress, ProgressColumn, Task, TaskID, TextColumn
from rich import get_console, reconfigure
from typing import override, Generator, Union, cast
from rich.text import Text
from rich.progress_bar import ProgressBar as _RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBar, MetricsTextColumn, CustomProgress, \
    ProcessingSpeedColumn


class RichProgressBarTheme:
    description: Union[str, "Style"] = "bright_blue"
    progress_bar: Union[str, "Style"] = "bright_green"
    progress_bar_finished: Union[str, "Style"] = "green"
    progress_bar_pulse: Union[str, "Style"] = "yellow"
    batch_progress: Union[str, "Style"] = "bright_blue"
    time: Union[str, "Style"] = "blue"
    processing_speed: Union[str, "Style"] = "blue"
    metrics: Union[str, "Style"] = "white"
    metrics_text_delimiter: str = " "
    metrics_format: str = ".4f"


class CustomRichProgressBar(RichProgressBar):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.theme = RichProgressBarTheme()

    @override
    def _init_progress(self, trainer: "L.Trainer") -> None:
        if self.is_enabled and (self.progress is None or self._progress_stopped):
            self._reset_progress_bar_ids()
            reconfigure(**self._console_kwargs)
            self._console = get_console()
            if hasattr(self._console, "_live_stack"):
                if len(self._console._live_stack) > 0:
                    self._console.clear_live()
            else:
                self._console.clear_live()
            self._metric_component = CustomMetricsTextColumn(
                trainer,
                self.theme.metrics,
                self.theme.metrics_text_delimiter,
                self.theme.metrics_format,
            )
            self.progress = CustomProgress(
                *self.configure_columns(trainer),
                self._metric_component,
                auto_refresh=False,
                disable=self.is_disabled,
                console=self._console,
            )
            self.progress.start()
            # progress has started
            self._progress_stopped = False

    @override
    def configure_columns(self, trainer: "L.Trainer") -> list:
        return [
            TextColumn("[progress.description]{task.description}"),
            CustomBarColumn(
                complete_style='bright_green',
                finished_style='bright_yellow',
                pulse_style='yellow',
            ),
            BatchesProcessedColumn(style=self.theme.batch_progress),
            CustomTimeColumn(style=self.theme.time),
            ProcessingSpeedColumn(style=self.theme.processing_speed),
        ]


# class CustomMetricsTextColumn(MetricsTextColumn):
#     def __init__(self,
#                  trainer: "L.Trainer",
#                  style: Union[str, "Style"],
#                  text_delimiter: str,
#                  metrics_format: str, ):
#         super().__init__(trainer, style, text_delimiter, metrics_format)
#
#     @override
#     def _generate_metrics_texts(self) -> Generator[str, None, None]:
#         for name, value in self._metrics.items():
#             if not isinstance(value, str):
#                 value = f"{value:{self._metrics_format}}"
#             yield f"{name}[{value}]"
#             # yield f"{Co.B}{name}[{Co.Y}{value}{Co.B}]{Co.RE}"

class CustomMetricsTextColumn(MetricsTextColumn):
    def __init__(self,
                 trainer: "L.Trainer",
                 style: Union[str, "Style"],
                 text_delimiter: str,
                 metrics_format: str, ):
        super().__init__(trainer, style, text_delimiter, metrics_format)

    @override
    def _generate_metrics_texts(self) -> Generator[Text, None, None]:
        for name, value in self._metrics.items():
            if not isinstance(value, str):
                value = f"{value:{self._metrics_format}}"
            metric_text = Text(style=self._style)
            metric_text.append(f"{name}[", style='bright_blue')
            metric_text.append(f"{value}", style='bright_yellow')
            metric_text.append(f"]", style='bright_blue')
            yield metric_text

    @override
    def render(self, task: "Task") -> Text:
        assert isinstance(self._trainer.progress_bar_callback, RichProgressBar)
        if (
                self._trainer.state.fn != "fit"
                or self._trainer.sanity_checking
                or self._trainer.progress_bar_callback.train_progress_bar_id != task.id
        ):
            return Text()

        if self._trainer.training and task.id not in self._tasks:
            self._tasks[task.id] = "None"
            if self._renderable_cache:
                self._current_task_id = cast(TaskID, self._current_task_id)
                self._tasks[self._current_task_id] = self._renderable_cache[self._current_task_id][1]
            self._current_task_id = task.id
        if self._trainer.training and task.id != self._current_task_id:
            return self._tasks[task.id]
        combined_text = Text(style=self._style)
        for metric_text in self._generate_metrics_texts():
            combined_text.append(metric_text)
            combined_text.append(self._text_delimiter)  # 添加分隔符
        return combined_text


class CustomBarColumn(BarColumn):
    """Overrides ``BarColumn`` to provide support for dataloaders that do not define a size (infinite size) such as
    ``IterableDataset``."""

    def render(self, task: "Task") -> _RichProgressBar:
        """Gets a progress bar widget for a task."""
        assert task.total is not None
        assert task.remaining is not None
        return _RichProgressBar(
            total=max(0, task.total),
            completed=max(0, task.completed),
            width=20,
            pulse=not task.started or not math.isfinite(task.remaining),
            animation_time=task.get_time(),
            style=self.style,
            complete_style=self.complete_style,
            finished_style=self.finished_style,
            pulse_style=self.pulse_style,
        )


class BatchesProcessedColumn(ProgressColumn):
    def __init__(self, style: Union[str, Style]):
        self.style = style
        super().__init__()

    def render(self, task: "Task") -> RenderableType:
        total = task.total if task.total != float("inf") else "--"
        return Text(f"{int(task.completed)}/{total}", style=self.style)


class CustomTimeColumn(ProgressColumn):
    # Only refresh twice a second to prevent jitter
    max_refresh = 0.5

    def __init__(self, style: Union[str, Style]) -> None:
        self.style = style
        super().__init__()

    @staticmethod
    def _format_time(seconds: int) -> str:
        """格式化时间，仅显示分钟和秒。"""
        if seconds is None:
            return "-:--"
        minutes, seconds = divmod(seconds, 60)
        return f"{minutes:02}:{seconds:02}"

    def render(self, task: "Task") -> Text:
        elapsed = task.finished_time if task.finished else task.elapsed
        remaining = task.time_remaining
        elapsed_delta = "-:--" if elapsed is None else self._format_time(int(elapsed))
        remaining_delta = "-:--" if remaining is None else self._format_time(int(remaining))
        return Text(f"{elapsed_delta}•{remaining_delta}", style=self.style)
