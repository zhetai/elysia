import click
import uvicorn


@click.group()
def cli():
    """Main command group for Elysia."""
    pass


@cli.command()
@click.option(
    "--port",
    default=8000,
    help="FastAPI Port",
)
@click.option(
    "--host",
    default="localhost",
    help="FastAPI Host",
)
@click.option(
    "--reload",
    default=True,
    help="FastAPI Reload",
)
def start(port, host, reload):
    """
    Run the FastAPI application.
    """
    uvicorn.run(
        "elysia.api.app:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    cli()
