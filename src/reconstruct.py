import os
from collections import defaultdict

import numpy as np
import uproot
from tqdm import tqdm

from reconstruction import config
from processing import event_selection
from reconstruction.objects import MET, Particle
from reconstruction.ttbar_dilepton import M_ELECTRON, M_MUON, reconstruct_event

if __name__ == "__main__":
    sm_path = os.path.join(
        "../data/mg5_data",
        f"{config.process_name}_{config.random_seed}",
        "Events/run_01_decayed_1/tag_1_delphes_events.root",
    )
    output_dir = f"../data/reconstructed_events/{config.process_name}_{config.random_seed}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    n_batches = 10

    print("Loading events...", end="\r")
    sm_events = uproot.open(sm_path)["Delphes"]
    print("Loading events...Done")

    print("Applying selection criteria...", end="\r")
    # Apply ATLAS selection criteria
    electron_mask = event_selection.select_electron(sm_events)
    muon_mask = event_selection.select_muon(sm_events)
    jets_mask = event_selection.select_jet(sm_events)

    # Get mask for b-jets
    bjets_mask = sm_events["Jet.BTag"].array()[jets_mask].astype(bool)

    # Select b-jets that pass selection criteria from Jet TTree
    bjets_mass = sm_events["Jet.Mass"].array()[jets_mask][bjets_mask]
    bjets_pt = sm_events["Jet.PT"].array()[jets_mask][bjets_mask]
    bjets_phi = sm_events["Jet.Phi"].array()[jets_mask][bjets_mask]
    bjets_eta = sm_events["Jet.Eta"].array()[jets_mask][bjets_mask]
    bjet = Particle(pt=bjets_pt, phi=bjets_phi, eta=bjets_eta, mass=bjets_mass)

    # Select electrons that pass selection criteria
    electron_pt = sm_events["Electron.PT"].array()[electron_mask]
    electron_phi = sm_events["Electron.Phi"].array()[electron_mask]
    electron_eta = sm_events["Electron.Eta"].array()[electron_mask]
    electron_charge = sm_events["Electron.Charge"].array()[electron_mask]
    electron = Particle(
        pt=electron_pt,
        phi=electron_phi,
        eta=electron_eta,
        mass=M_ELECTRON,
        charge=electron_charge,
    )

    # Select muons that pass selection criteria
    muon_pt = sm_events["Muon.PT"].array()[muon_mask]
    muon_phi = sm_events["Muon.Phi"].array()[muon_mask]
    muon_eta = sm_events["Muon.Eta"].array()[muon_mask]
    muon_charge = sm_events["Muon.Charge"].array()[muon_mask]
    muon = Particle(
        pt=muon_pt, phi=muon_phi, eta=muon_eta, mass=M_MUON, charge=muon_charge
    )

    # MET for all events
    met_magnitude = sm_events["MissingET.MET"].array()
    met_phi = sm_events["MissingET.Phi"].array()
    met = MET(magnitude=met_magnitude, phi=met_phi)

    print("Applying selection criteria...Done")

    step_size = len(muon_phi) // n_batches
    rng = np.random.default_rng(config.random_seed)
    for batch_idx in tqdm(range(n_batches)):
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
            for idx in tqdm(range(init_idx, end_idx), leave=False)
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
                os.path.join(output_dir, f"{name}_batch_{batch_idx}.npy"), "wb"
            ) as f:
                np.save(f, p_array)
        del recos, reco_arrays, reconstructed_events
