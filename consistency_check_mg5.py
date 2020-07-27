import os
import typer
import uproot
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

from processing import counts
from plotting import histos


def checks(root_path: Path = typer.Argument(..., help="Path to .root file with Delphes events"),
           show: bool = typer.Option(True, help="Show popup windows with plots"),
           plots_path: Path = typer.Argument(None, help="Path to save plots")):

    if not root_path.exists():
        typer.secho("\nWrong path for .root file", fg=typer.colors.RED, bold=True)
        raise typer.Abort()

    if (plots_path is not None) and plots_path.exists():
        confirm_message = typer.style(
            "\nDirectory for plots already exists. Rewrite?",
            fg=typer.colors.YELLOW,
            bold=True
        )
        _ = typer.confirm(confirm_message, abort=True)
    elif (plots_path is not None) and not plots_path.exists():
        typer.secho("\nCreating directory for plots", fg=typer.colors.GREEN, bold=True)
        os.mkdir(plots_path)

    events = uproot.open(root_path)["Delphes"]

    # Number of Jets
    n_jets = counts.n_particles(events, "Jet.BTag")
    fig = histos.hist_n_particles(n_jets, "N Jets")
    if show:
        plt.show()
    if plots_path:
        fig.savefig(os.path.join(plots_path, "n_jets.png"))

    # Number of b-Jets
    n_bjets = counts.n_particles_from_tag(events, "Jet.BTag")
    fig = histos.hist_n_particles(n_bjets, "N b-Jets")
    if show:
        plt.show()
    if plots_path:
        fig.savefig(os.path.join(plots_path, "n_bjets.png"))

    # Number of electrons
    n_elecs = counts.n_particles(events, "Electron.PT")
    fig = histos.hist_n_particles(n_elecs, "N Electrons")
    if show:
        plt.show()
    if plots_path:
        fig.savefig(os.path.join(plots_path, "n_elecs.png"))

    # Number of muons
    n_muons = counts.n_particles(events, "Muon.PT")
    fig = histos.hist_n_particles(n_muons, "N Muons")
    if show:
        plt.show()
    if plots_path:
        fig.savefig(os.path.join(plots_path, "n_muons.png"))

    # Number of Leptons
    n_leps = np.array(n_elecs) + np.array(n_muons)
    fig = histos.hist_n_particles(n_leps, "N Leptons")
    if show:
        plt.show()
    if plots_path:
        fig.savefig(os.path.join(plots_path, "n_leptons.png"))


if __name__ == "__main__":
    typer.run(checks)
