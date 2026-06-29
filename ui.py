"""Terminal UI — all rich calls live here, orchestrator stays clean."""

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

console = Console()


def get_log_handler():
    return RichHandler(console=console, rich_tracebacks=True, show_path=False)


def print_banner(container, data_dir):
    console.print(Panel.fit(
        f"[bold cyan]OpenModelica Workflow Orchestrator[/bold cyan]\n"
        f"[dim]Container:[/dim] {container}\n"
        f"[dim]Data dir: [/dim] {data_dir}",
        border_style="cyan",
    ))


def print_rule(title):
    console.print(Rule(f"[bold]{title}[/bold]", style="blue"))


def print_cmd(cmd):
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]\n")


def print_stage_result(success, duration):
    if success:
        console.print(f"[bold green]✓[/bold green] Completed in {duration}\n")
    else:
        console.print(f"[bold red]✗[/bold red] Failed after {duration}\n")


def print_summary(results):
    table = Table(title="Workflow Summary", show_header=True, header_style="bold cyan")
    table.add_column("Stage", style="dim", width=40)
    table.add_column("Status", justify="center", width=10)
    table.add_column("Duration", justify="right", width=12)

    for r in results:
        status = Text("✓ OK", style="bold green") if r["success"] else Text("✗ FAIL", style="bold red")
        table.add_row(r["description"], status, r["duration"])

    console.print()
    console.print(table)


def print_workflow_success(total):
    console.print(Panel(
        f"[bold green]All stages completed successfully[/bold green]\n"
        f"[dim]Total time: {total}[/dim]",
        border_style="green",
    ))


class Spinner:
    """Context manager — shows a spinner while a stage runs."""

    def __init__(self, description):
        self._description = description
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        )

    def __enter__(self):
        self._progress.__enter__()
        self._progress.add_task(f"[cyan]{self._description}…", total=None)
        return self

    def __exit__(self, *args):
        self._progress.__exit__(*args)