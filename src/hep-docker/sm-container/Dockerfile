FROM ubuntu:20.04

USER root
WORKDIR /root

SHELL [ "/bin/bash", "-c" ]

ARG PYTHON_VERSION_TAG=3.10.5
ARG LINK_PYTHON_TO_PYTHON3=1

# Existing lsb_release causes issues with modern installations of Python3
# https://github.com/pypa/pip/issues/4924#issuecomment-435825490
# Set (temporarily) DEBIAN_FRONTEND to avoid interacting with tzdata
RUN apt-get -qq -y update && \
    DEBIAN_FRONTEND=noninteractive apt-get -qq -y install \
        gcc \
        g++ \
        gfortran \
        zlibc \
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
        bc && \
    mv /usr/bin/lsb_release /usr/bin/lsb_release.bak && \
    apt-get -y autoclean && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/*

COPY install_python.sh install_python.sh
RUN bash install_python.sh ${PYTHON_VERSION_TAG} ${LINK_PYTHON_TO_PYTHON3} && \
    rm -r install_python.sh Python-${PYTHON_VERSION_TAG}

RUN python -m pip --no-cache-dir install --upgrade pip setuptools wheel && \
    python -m pip --no-cache-dir install six numpy

# Install ROOT
RUN cd /opt && \
    mkdir root && \
    cd root && \
    wget https://root.cern/download/root_v6.26.06.Linux-ubuntu20-x86_64-gcc9.4.tar.gz && \
    tar -xzf root_v6.26.06.Linux-ubuntu20-x86_64-gcc9.4.tar.gz --strip=1

# Install MadGraph 5 and pythia8 (through MadGraph5)
ARG MG_VERSION=3.4.0
RUN cd /opt && \
    mkdir MG5_aMC && \
    cd MG5_aMC && \wget --quiet https://launchpad.net/mg5amcnlo/3.0/3.4.x/+download/MG5_aMC_v${MG_VERSION}.tar.gz && \ 
    tar -xzf MG5_aMC_v${MG_VERSION}.tar.gz --strip=1 && \
    rm MG5_aMC_v${MG_VERSION}.tar.gz && \
    printf '\nexport PATH=/opt/MG5_aMC/bin:"${PATH}"\n' >> /root/.bashrc

ENV ROOTSYS=/opt/root
ENV PATH=$ROOTSYS/bin:$PATH
ENV PYTHONPATH=$ROOTSYS/lib:$PYTHONPATH
ENV LD_LIBRARY_PATH=$ROOTSYS/lib:$LD_LIBRARY_PATH
ENV PYTHIA8=/opt/MG5_aMC/pythia8

ENTRYPOINT ["/bin/bash", "-l", "-c"]
CMD ["/bin/bash"]