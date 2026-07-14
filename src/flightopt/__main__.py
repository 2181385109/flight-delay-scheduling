"""Enable ``python -m flightopt ...`` to invoke the Typer CLI."""

from flightopt.cli import app

if __name__ == "__main__":
    app()
