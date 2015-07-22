FROM      ubuntu:latest
MAINTAINER Michael D'Amour <mdamour@pentaho.com>

# make sure the package repository is up to date
RUN sed -i.bak 's/main$/main universe/' /etc/apt/sources.list
RUN apt-get update
RUN apt-get upgrade -y

RUN apt-get -y -q install python-software-properties software-properties-common
RUN locale-gen en_US en_US.UTF-8
RUN update-locale LANG=en_US.UTF-8 LC_MESSAGES=POSIX

RUN apt-get -y -q install libboost-all-dev g++
RUN apt-get -y -q install unzip nano git python wget curl expect

RUN git clone --depth 5 https://github.com/peter-ch/MultiNEAT.git
#RUN cd MultiNEAT; git checkout f616390889d92b8674a2e8fee8d2b0d7c6badcae; 
RUN cd MultiNEAT; git checkout 9232a56d4526a0bd69a2fce44f656f5331009bb8
RUN cd MultiNEAT; python setup.py install

#ENTRYPOINT bash
ENTRYPOINT git clone --depth 1 https://github.com/brosander/lc-tools.git; cd /lc-tools; python lcNeat.py -i /historical-data -o /historical-data/output
