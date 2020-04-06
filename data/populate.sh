#!/bin/bash
wget http://www.topology-zoo.org/files/archive.zip
unzip archive.zip -d topologyzoo

wget http://totem.run.montefiore.ulg.ac.be/files/data/traffic-matrices-anonymized-v2.tar.bz2
mkdir totem
tar xjf traffic-matrices-anonymized-v2.tar.bz2 -C totem
