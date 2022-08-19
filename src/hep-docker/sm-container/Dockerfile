FROM alan200276/ubuntu:HEPtools


FROM continuumio/miniconda3:latest

RUN apt-get -qq -y update && \
    DEBIAN_FRONTEND=noninteractive apt-get -qq -y install \
        gcc \
        g++ \
        gfortran \
        zlib1g-dev \
        libssl-dev \
        libx11-dev \
        libxpm-dev \
        libxft-dev \
        libxext-dev \
        libssl-dev \
        libbz2-dev \
        libsqlite3-dev \
        libncurses5-dev \
        libgdbm-dev \
        libgdbm-compat-dev \
        liblzma-dev \
        libreadline-dev \
        uuid-dev \
        libffi-dev \
        tk-dev \
        wget \
        curl \
        git \
        make \
        cmake \
        sudo \
        bash-completion \
        tree \
        vim \
        software-properties-common \
        bc

COPY --from=0 /root/Delphes-3.4.2 /root/Delphes-3.4.2
COPY --from=0 /root/MG5_aMC_v2_7_3 /root/MG5_aMC_v2_7_3
COPY --from=0 /root/root_py3 /root/root_py3

RUN conda create -n mg5 python=2.7 

RUN printf '\nexport PATH=/root/MG5_aMC_v2_7_3/bin:"${PATH}"\n' >> /root/.bashrc

# COPY ./optimal_observables-0.1-py3-none-any.whl .
# RUN pip install ./optimal_observables-0.1-py3-none-any.whl