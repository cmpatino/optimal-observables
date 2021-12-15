import os

base_path = "../data/reconstructed_events/"
pos_process = "SM_spin-ON"
neg_process = "SM_spin-OFF"

pos_reco_paths = [
    os.path.join(base_path, path)
    for path in os.listdir(base_path) if pos_process in path
]
neg_reco_paths = [
    os.path.join(base_path, path)
    for path in os.listdir(base_path) if neg_process in path
]

dataset_config = {
    "pos_reconstruction_paths": pos_reco_paths,
    "neg_reconstruction_paths": neg_reco_paths,
    "only_cosine_terms": True,
}
