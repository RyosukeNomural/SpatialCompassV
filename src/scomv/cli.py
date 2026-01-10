"""Console script for scomv."""

import typer
from rich.console import Console
import scomv
from scomv import utils

app = typer.Typer()
console = Console()
@app.command()
def version():
    """Show scomv version."""
    print(scomv.__version__)


@app.command()
def main():
    """Console script for scomv."""
    console.print("Replace this message by putting your code into "
               "scomv.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    utils.do_something_useful()


if __name__ == "__main__":
    app()
