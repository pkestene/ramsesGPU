#!/bin/bash
#MSUB -r dumpMpiInfo
#MSUB -n 8
#MSUB -T 600
#MSUB -q test
#MSUB -o dumpMpiInfo_%I.o
#MSUB -e dumpMpiInfo_%I.e
#MSUB -A paxxxx                         # change here to the right project Id
##MSUB -@ pierre.kestener@cea.fr:end
set -x
cd ${BRIDGE_MSUB_PWD}
ccc_mprun ./dumpMpiInfo -fname test.txt


