# Event Generation Commands

1. Run Container
    1. sudo docker run -it hep-sw

2. Open MadGraph5
    1. mg5_aMC

3. Run event generation for SM
    1. `generate p p > t t~`
    2. `launch -m`
    3. Activate Pythia, Delphes, and MadSpin
    4. Set quark mass to 172.5 GeV on param card.
    5. Set seed and number of events on run card.
    6. Change madspin card (seed and decays).

3. Run event generation for EFT
    1. `import model dim6top_LO_UFO`
    2. `generate p p > t t~`
    3. `launch -m`
    4. Activate Pythia, Delphes, and MadSpin
    5. Set quark mass to 172.5 GeV on param card.
    6. Set seed and number of events on run card
    7. Change madspin card (seed and decays).

4. Save events to host
    1. sudo docker container ls
    2. sudo docker cp <containerId>:/file/path/in/container/file /host/local/path/file
        + File of interest is the one in `Events/`

# Publish container to GitHub Registry

## Setup of GitHub Credentials

These steps are required once.

1. [Generate PAT](https://docs.github.com/en/github/authenticating-to-github/keeping-your-account-and-data-secure/creating-a-personal-access-token) on GitHub with permissions to read and write packages.

2. Login into the Container registry service at ghcr.io using `docker login ghcr.io -u USERNAME`. Replace USERNAME with your GitHub username and use the PAT of step 1 as a password.

## Building and Pushing the Container

+ Build container: `docker build -t ghcr.io/cmpatino/hep-software:latest .`
+ Publish image: `docker push ghcr.io/cmpatino/hep-software:latest`