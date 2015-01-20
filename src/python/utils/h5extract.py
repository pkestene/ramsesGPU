"""
   Extract dataset from a HDF55, and write results in another HDF5 file

"""
from __future__ import print_function

import sys
import numpy
import h5py

def extract(inputFilename, datasetName, outputFilename):
    print("Trying to extract dataset %s from file %s" % (datasetName,inputFilename) )
    iFile=h5py.File(inputFilename, 'r')

    if datasetName in iFile:
        # open dataset
        data = iFile[datasetName]

        # if output does not already exist, copy dataset
        if not h5py.is_hdf5(outputFilename):
            oFile=h5py.File(outputFilename, 'w')
            oFile.create_dataset(datasetName, data=data.value)
            oFile.close()
        else:
            print("output file %s already exists ! Delete file before re-running" % outputFilename)
    else:
        print("datasetName : %s is not in input HDF5 file !" % datasetName)
        print("Do not write any output file...")

    iFile.close()


if __name__ == '__main__':
    if (len(sys.argv)>=4):
        extract(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Not enough arguments !")
        print("Try:")
        print("python h5extract input.h5 datasetName output.h5")
