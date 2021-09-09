import os

import numpy as np
import uproot
from tqdm import tqdm

from reconstruction.ttbar_dilepton import reconstruct_event
from processing import event_selection


if __name__ == "__main__":
    process_name = "SM_spin-OFF_100k"
    random_seed = 0

    sm_path = os.path.join(
        "../data/mg5_data",
        f"{process_name}_{random_seed}",
        "Events/run_01_decayed_1/tag_1_delphes_events.root",
    )
    output_dir = f"../reconstructed_events/{process_name}_{random_seed}"
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

    # Select electrons that pass selection criteria
    electron_pt = sm_events["Electron.PT"].array()[electron_mask]
    electron_phi = sm_events["Electron.Phi"].array()[electron_mask]
    electron_eta = sm_events["Electron.Eta"].array()[electron_mask]
    electron_charge = sm_events["Electron.Charge"].array()[electron_mask]

    # Select muons that pass selection criteria
    muon_pt = sm_events["Muon.PT"].array()[muon_mask]
    muon_phi = sm_events["Muon.Phi"].array()[muon_mask]
    muon_eta = sm_events["Muon.Eta"].array()[muon_mask]
    muon_charge = sm_events["Muon.Charge"].array()[muon_mask]

    # MET for all events
    met = sm_events["MissingET.MET"].array()
    met_phi = sm_events["MissingET.Phi"].array()
    print("Applying selection criteria...Done")

    reco_names = [
        "p_top",
        "p_l_t",
        "p_b_t",
        "p_nu_t",
        "p_tbar",
        "p_l_tbar",
        "p_b_tbar",
        "p_nu_tbar",
        "idx",
        "weight",
    ]
    step_size = len(muon_phi) // n_batches
    rng = np.random.default_rng(random_seed)
    for batch_idx in tqdm(range(n_batches)):
        init_idx = batch_idx * step_size
        end_idx = init_idx + step_size
        reconstructed_events = [
            reconstruct_event(
                bjets_mass=bjets_mass[idx],
                bjets_pt=bjets_pt[idx],
                bjets_phi=bjets_phi[idx],
                bjets_eta=bjets_eta[idx],
                electron_pt=electron_pt[idx],
                electron_phi=electron_phi[idx],
                electron_eta=electron_eta[idx],
                electron_charge=electron_charge[idx],
                muon_pt=muon_pt[idx],
                muon_phi=muon_phi[idx],
                muon_eta=muon_eta[idx],
                muon_charge=muon_charge[idx],
                met=met[idx],
                met_phi=met_phi[idx],
                idx=idx,
                rng=rng,
            )
            for idx in tqdm(range(init_idx, end_idx), leave=False)
        ]

        recos = {name: [] for name in reco_names}

        for event in reconstructed_events:
            if event is None:
                continue
            for name, reco_p in zip(reco_names, event):
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
