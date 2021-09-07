FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -yq \
    cmake \
    curl \
    g++ \
    gfortran \
    git \
    graphviz-dev \
    libcfitsio-dev \
    libfftw3-dev \
    libftgl-dev \
    libglew-dev \
    libglu1-mesa-dev \
    libgsl-dev \
    libjpeg-dev \
    libkrb5-dev \
    libldap2-dev \
    libmysqlclient-dev \
    libpcre3-dev \
    libpng-dev \
    libqt4-dev \
    libqt4-opengl-dev \
    libssl-dev \
    libtbb-dev \
    libx11-dev \
    libxext-dev \
    libxft-dev \
    libxi-dev \
    libxml2-dev \
    libxmu-dev \
    libxpm-dev \
    libxt-dev \
    make \
    python-dev \
    rsync \
    tcl \
    wget \
    vim

# Install ROOT
RUN mkdir -p /opt \
 && curl -sL https://root.cern/download/root_v6.18.00.Linux-ubuntu18-x86_64-gcc7.4.tar.gz | tar -C /opt -zxf -

#Install MadGraph 5 and pythia8 (through MadGraph5)
RUN curl -sL https://launchpad.net/mg5amcnlo/2.0/2.7.x/+download/MG5_aMC_v2.7.3.tar.gz | tar -C /opt -zxf - \
     && echo "install pythia8\nexit" | /opt/MG5_aMC_v2_7_3/bin/mg5_aMC \
     && export LD_LIBRARY_PATH=/home/carlosmiguelpatino/Software/MadGraph5/HEPTools/lhapdf6_py3/lib

ENV ROOTSYS=/opt/root
ENV PATH=$ROOTSYS/bin:$PATH
ENV PYTHONPATH=$ROOTSYS/lib:$PYTHONPATH
ENV LD_LIBRARY_PATH=$ROOTSYS/lib:$LD_LIBRARY_PATH
ENV PYTHIA8=/opt/MG5_aMCv2_7_3/pythia8

# Install Delphes
RUN git clone https://github.com/delphes/delphes.git /opt/MG5_aMC_v2_7_3/Delphes \
    && cd /opt/MG5_aMC_v2_7_3/Delphes \
    && make -j4


RUN echo alias mg5_aMC=\"/opt/MG5_aMC_v2_7_3/bin/mg5_aMC\" >> ~/.bash_aliases

RUN mkdir /generated_events

COPY madspin_card_default.dat /opt/MG5_aMC_v2_7_3/Template/Common/Cards/madspin_card_default.dat