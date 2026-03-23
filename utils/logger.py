"""
Rich Console Logger for Barista CI Tool.

Provides beautiful, detailed console output across all agents using the Rich library.
Centralized logging with consistent styling, panels, tables, and progress tracking.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.columns import Columns
from rich.rule import Rule
from rich import box
from contextlib import contextmanager
import time

# Global console instance
console = Console()


# ─── Workflow Phase Banners ───

def banner(title: str, subtitle: str = "", style: str = "bold cyan"):
    """Display a major workflow phase banner."""
    content = f"[bold]{title}[/bold]"
    if subtitle:
        content += f"\n[dim]{subtitle}[/dim]"
    console.print(Panel(content, style=style, box=box.DOUBLE, expand=True))


def section(title: str, icon: str = "🔹"):
    """Display a section header within a phase."""
    console.print(Rule(f" {icon} {title} ", style="blue"))


# ─── Status Messages ───

def info(msg: str, icon: str = "ℹ️"):
    console.print(f"  {icon}  {msg}")


def success(msg: str, icon: str = "✅"):
    console.print(f"  {icon}  [green]{msg}[/green]")


def warning(msg: str, icon: str = "⚠️"):
    console.print(f"  {icon}  [yellow]{msg}[/yellow]")


def error(msg: str, icon: str = "❌"):
    console.print(f"  {icon}  [bold red]{msg}[/bold red]")


def detail(msg: str, indent: int = 6):
    console.print(f"{' ' * indent}[dim]{msg}[/dim]")


def step(num: int, total: int, msg: str):
    """Step indicator like [2/5] Doing something..."""
    console.print(f"  [cyan][{num}/{total}][/cyan] {msg}")


# ─── Tables ───

def provider_table(providers: list):
    """Display active search providers in a table."""
    table = Table(
        title="🔎 Active Search Providers",
        box=box.ROUNDED,
        show_lines=False,
        title_style="bold blue",
    )
    table.add_column("Provider", style="bold white", min_width=10)
    table.add_column("Status", style="green", min_width=8)

    for name in providers:
        table.add_row(name, "● Active")

    console.print(table)


def article_table(articles: list, title: str = "Articles"):
    """Display articles in a compact table."""
    table = Table(
        title=title,
        box=box.SIMPLE_HEAVY,
        show_lines=False,
        title_style="bold",
        expand=True,
    )
    table.add_column("#", style="cyan", width=3)
    table.add_column("Score", style="bold yellow", width=7)
    table.add_column("Title", style="white", min_width=30, max_width=55, no_wrap=True)
    table.add_column("Domain", style="dim", width=20)
    table.add_column("Type", style="magenta", width=8)

    for i, article in enumerate(articles[:15], 1):
        score = getattr(article, "score", 0.0) or 0.0
        score_str = f"{score:.3f}" if score > 0 else "—"
        title_text = (article.title[:52] + "...") if len(article.title) > 55 else article.title
        domain = getattr(article, "domain", "—") or "—"
        stype = getattr(article, "source_type", "—") or "—"
        table.add_row(str(i), score_str, title_text, domain, stype)

    console.print(table)


def score_table(scored_items: list, title: str = "Hybrid Scoring Results"):
    """Display fuzzy + weighted scoring results."""
    table = Table(
        title=title,
        box=box.SIMPLE_HEAVY,
        title_style="bold yellow",
        expand=True,
    )
    table.add_column("#", style="cyan", width=3)
    table.add_column("Hybrid", style="bold green", width=8)
    table.add_column("Fuzzy", style="yellow", width=8)
    table.add_column("Weighted", style="blue", width=9)
    table.add_column("Title", style="white", min_width=30, max_width=50, no_wrap=True)
    table.add_column("Verdict", width=10)

    for i, (hybrid, fuzzy, weighted, title_text, passed) in enumerate(scored_items[:15], 1):
        verdict = "[green]✓ Pass[/green]" if passed else "[red]✗ Reject[/red]"
        t = (title_text[:47] + "...") if len(title_text) > 50 else title_text
        table.add_row(
            str(i),
            f"{hybrid:.3f}",
            f"{fuzzy:.3f}",
            f"{weighted:.3f}",
            t,
            verdict,
        )

    console.print(table)


def merge_summary(raw_total: int, after_url: int, after_sim: int):
    """Display merge/dedup statistics."""
    table = Table(box=box.ROUNDED, show_header=False, title_style="bold")
    table.add_column("Stage", style="bold white", width=20)
    table.add_column("Count", style="cyan", width=8)
    table.add_column("Removed", style="red", width=10)

    table.add_row("Raw merged", str(raw_total), "—")
    table.add_row("After URL dedup", str(after_url), f"-{raw_total - after_url}")
    table.add_row("After sim dedup", str(after_sim), f"-{after_url - after_sim}")

    console.print(Panel(table, title="📦 Merge Pipeline", border_style="blue"))


def report_summary(report: dict):
    """Display a beautiful report summary panel."""
    title = report.get("report_title", "Research Report")

    findings = report.get("key_findings", [])
    official = report.get("official_insights", [])
    trusted = report.get("trusted_insights", [])
    refs = report.get("references", [])

    stats = Table(box=None, show_header=False, padding=(0, 2))
    stats.add_column("Metric", style="dim")
    stats.add_column("Value", style="bold cyan")
    stats.add_row("Key Findings", str(len(findings)))
    stats.add_row("Official Insights", str(len(official)))
    stats.add_row("Trusted Insights", str(len(trusted)))
    stats.add_row("References", str(len(refs)))

    content = Text()
    exec_sum = report.get("executive_summary", "")
    if exec_sum:
        preview = exec_sum[:300] + "..." if len(exec_sum) > 300 else exec_sum
        content.append(preview)

    console.print(Panel(
        Columns([stats, content], expand=True),
        title=f"📋 {title}",
        border_style="green",
        box=box.DOUBLE,
    ))


# ─── Progress Tracking ───

@contextmanager
def phase_progress(description: str):
    """Context manager for a phase with elapsed time."""
    start = time.time()
    console.print(f"\n  ⏳ [bold]{description}[/bold]...")
    yield
    elapsed = time.time() - start
    console.print(f"  ⏱️  [dim]Completed in {elapsed:.1f}s[/dim]")


def get_progress():
    """Get a Rich progress bar for multi-step operations."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("[cyan]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )
