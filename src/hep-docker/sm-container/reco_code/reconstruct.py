import os
from collections import defaultdict

import awkward as ak
import numpy as np
import uproot
from rich.console import Console
from tqdm import tqdm

from reco_config import root_file_path, recos_output_dir, random_seed
import event_selection
from objects import MET, Particle
from ttbar_dilepton import M_ELECTRON, M_MUON, reconstruct_event

if __name__ == "__main__":
    console = Console()
    if not os.path.exists(recos_output_dir):
        os.makedirs(recos_output_dir)
    else:
        raise ValueError(f"{recos_output_dir} already exists. Stopping to avoid overwriting.")
    n_batches = 10

    console.print("Loading events...", style="bold yellow")
    sm_events = uproot.open(root_file_path)["Delphes"]
    console.print("Loading events...Done\n", style="bold green")

    console.print("Applying selection criteria...", style="bold yellow")
    electron_mask = event_selection.select_electron(sm_events)
    muon_mask = event_selection.select_muon(sm_events)
    jets_mask = event_selection.select_jet(sm_events)

    # Get mask for b-jets
    bjets_tags = sm_events["Jet.BTag"].array()[jets_mask]
    bjets_mask = ak.values_astype(bjets_tags, bool)

    # Select b-jets that pass selection criteria from Jet TTree
    bjets_mass = sm_events["Jet.Mass"].array()[jets_mask][bjets_mask]
    bjets_pt = sm_events["Jet.PT"].array()[jets_mask][bjets_mask]
    bjets_phi = sm_events["Jet.Phi"].array()[jets_mask][bjets_mask]
    bjets_eta = sm_events["Jet.Eta"].array()[jets_mask][bjets_mask]
    bjet = Particle(
        pt=[np.array(event) for event in bjets_pt],
        phi=[np.array(event) for event in bjets_phi],
        eta=[np.array(event) for event in bjets_eta],
        mass=[np.array(event) for event in bjets_mass],
    )

    # Select electrons that pass selection criteria
    electron_pt = sm_events["Electron.PT"].array()[electron_mask]
    electron_phi = sm_events["Electron.Phi"].array()[electron_mask]
    electron_eta = sm_events["Electron.Eta"].array()[electron_mask]
    electron_charge = sm_events["Electron.Charge"].array()[electron_mask]
    electron = Particle(
        pt=[np.array(event) for event in electron_pt],
        phi=[np.array(event) for event in electron_phi],
        eta=[np.array(event) for event in electron_eta],
        mass=M_ELECTRON,
        charge=[np.array(event) for event in electron_charge],
    )

    # Select muons that pass selection criteria
    muon_pt = sm_events["Muon.PT"].array()[muon_mask]
    muon_phi = sm_events["Muon.Phi"].array()[muon_mask]
    muon_eta = sm_events["Muon.Eta"].array()[muon_mask]
    muon_charge = sm_events["Muon.Charge"].array()[muon_mask]
    muon = Particle(
        pt=[np.array(event) for event in muon_pt],
        phi=[np.array(event) for event in muon_phi],
        eta=[np.array(event) for event in muon_eta],
        mass=M_MUON,
        charge=[np.array(event) for event in muon_charge],
    )

    # MET for all events
    met_magnitude = sm_events["MissingET.MET"].array()
    met_phi = sm_events["MissingET.Phi"].array()
    met = MET(
        magnitude=[np.array(event) for event in met_magnitude],
        phi=[np.array(event) for event in met_phi],
    )

    console.print("Applying selection criteria...Done\n", style="bold green")

    console.print("Reconstructing events...", style="bold yellow")
    step_size = len(muon_phi) // n_batches
    rng = np.random.default_rng(random_seed)
    for batch_idx in tqdm(range(n_batches), desc="Reconstructing events"):
        init_idx = batch_idx * step_size
        end_idx = init_idx + step_size
        reconstructed_events = [
            reconstruct_event(
                bjet=bjet,
                electron=electron,
                muon=muon,
                met=met,
                idx=idx,
                rng=rng,
            )
            for idx in tqdm(range(init_idx, end_idx), leave=False, desc="Resconstructing batch")
        ]

        recos = defaultdict(list)

        for event in reconstructed_events:
            if event is None:
                continue
            for name, reco_p in event.return_values().items():
                recos[name].append(reco_p.reshape(1, -1))

        reco_arrays = {
            name: np.concatenate(reco_list, axis=0) for name, reco_list in recos.items()
        }

        for name, p_array in reco_arrays.items():
            with open(
                os.path.join(recos_output_dir, f"{name}_batch_{batch_idx}.npy"), "wb"
            ) as f:
                np.save(f, p_array)
        del recos, reco_arrays, reconstructed_events
    console.print("Reconstructing events...Done", style="bold green")
