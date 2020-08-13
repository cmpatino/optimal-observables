# Consistency Checks: Generation

## Standard Model: Spin On vs Spin Off

:heavy_check_mark: There should be a difference in the delpha phi plot similar to Powheg (C=0) and the data in [this](https://atlas.cern/sites/atlas-public.web.cern.ch/files/fig2top.png) plot.

![dphi](./spin_off_vs_on/dphi_leptons.png)

:heavy_check_mark: Number of jets peaks at 2

![n_jets](./spin_off_vs_on/n_jets.png)
![n_leptons](./spin_off_vs_on/n_leptons.png)

:heavy_check_mark: Mass centered around 346 GeV

![mtt](./spin_off_vs_on/m_tt.png)
![pt](./spin_off_vs_on/pt_tt.png)

## SM vs EFT (ctG=0)

The discrepancy of the top-antitop mass is caused by the change of the [EFT implementation](https://arxiv.org/pdf/1802.07237.pdf): instead of 173 GeV they set it to 172 GeV.

![mtt](./sm_vs_ctG0/m_tt.png)

:heavy_check_mark: All plots should be the same. 

![dphi](./sm_vs_ctG0/dphi_leptons.png)

:heavy_check_mark: Number of jets peaks at 2

![n_jets](./sm_vs_ctG0/n_jets.png)
![n_leptons](./sm_vs_ctG0/n_leptons.png)
![pt](./sm_vs_ctG0/pt_tt.png)

## Different Values for ctG

+ There are slight shifts in the distributions (observed better comparing ctG=0 and ctG=5).

![dphi](./ctg_sweep/dphi_leptons.png)
![pt](./ctg_sweep/pt_tt.png)

:heavy_check_mark: Number of jets peaks at 2

![n_jets](./ctg_sweep/n_jets.png)

:heavy_check_mark: Mass remains consistently centered at 344 GeV

![m_tt](./ctg_sweep/m_tt.png)
![n_jets](./ctg_sweep/n_jets_all.png)
![dphi](./ctg_sweep/dphi_leptons_all.png)