# 1. Run Container
1. sudo docker run -it hep-sw

# 2. Open MadGraph5
1. mg5_aMC

# 3a. Run event generation for SM
1. generate p p > t t~
2. launch -m
3. Activate Pythia, Delphes, and MadSpin
4. Set quark mass to 172.5 GeV
5. Change madspin card.

# 3b. Run event generation for EFT
1. import model dim6top_LO_UFO
2. generate p p > t t~
3. launch -m
4. Activate Pythia, Delphes, and MadSpin
5. Set quark mass to 172.5 GeV
6. Change madspin card with relevant commands

# 4. Save events to host
1. docker cp <containerId>:/file/path/in/container/file /host/local/path/file
    + File of interest is the one in `Events/`