"""
This file is run first and extracts the relevant fields of the small_pub_.vtu file
"""

import sys, os
import numpy as np
import pyvista as pv
import eofs
from eofs.standard import Eof
import variables
import time

class extractFieldsAndPCA():
    def __init__(self, directory_data, start, end, start_pca, end_pca, nsize):

        self.directory_data = directory_data
        self.start = start
        self.end = end
        self.nsize = nsize

        # For PCA analysis specific time-steps
        self.start_pca = start_pca
        self.end_pca = end_pca

    def extractFields(self):
        tracer_data = np.zeros((self.end-self.start, self.nsize))
        vel_data = np.zeros((self.end-self.start, self.nsize))
        k = 0
        for i in np.arange(self.start, self.end):
                filename = self.directory_data + 'Small_pub_' + str(i) + '.vtu'
                mesh = pv.read(filename)
                tracer_data[k, :] = np.squeeze(mesh.get_array('PassiveTracer_CO2'))
                vel_data[k, :] = mesh.get_array('Temperature')
                print(k)
                k = k + 1


        np.save(self.directory_data + 'Tracer_data_' + str(self.start) + '_to_' + str(self.end), tracer_data)
        np.save(self.directory_data + 'Velocity_data_' + str(self.start) + '_to_' + str(self.end), vel_data)

    def PCA(self, field_name):
        field_name = field_name
        start_interv = self.start_pca
        end_interv = self.end_pca
        observationPeriod = 'data_' + str(start_interv) + '_to_' + str(end_interv)
        modelData = np.load(self.directory_data + '' + field_name + '_' + observationPeriod + '.npy')

        # Velocity is a 3D vector and needs to be reshaped before the PCA
        if 'Velocity' in field_name:
            modelData = np.reshape(modelData, (modelData.shape[0], modelData.shape[1] * modelData.shape[2]), order='F')

        # Standardise the data with mean 0
        meanData = np.nanmean(modelData, 0)
        stdData = np.nanstd(modelData)
        modelDataScaled = (modelData - meanData) / stdData

        #PCA solver
        solver = Eof(modelDataScaled)

        # Principal Components time-series
        pcs = solver.pcs()
        # Projection
        eof = solver.eofs()
        # Cumulative variance
        varianceCumulative = np.cumsum(solver.varianceFraction())

        np.save(self.directory_data + '/' + 'pcs_' + field_name + '_' + observationPeriod,
                pcs)
        np.save(self.directory_data + '/' + 'eofs_' + field_name + '_' + observationPeriod,
                eof)
        np.save(self.directory_data + '/' + 'varCumulative_' + field_name + '_' + observationPeriod,
                varianceCumulative)
        np.save(self.directory_data + '/' + 'mean_' + field_name + '_' + observationPeriod,
                meanData)
        np.save(self.directory_data + '/' + 'std_' + field_name + '_' + observationPeriod,
                stdData)

if __name__ == '__main__':

    directory_data = 'G:/PHD_FILES_EXTERNAL/pub/COVID_ROM_ACH_2_5_heating_P0DG/'
    field_name = 'Tracer'

    # Interval within the simulation for extracting data
    start = 1 
    end = 680
    # Interval within the extracted data to perform PCA
    start_pca = 1
    end_pca = 680
    # Number of nodes in the unstructured mesh
    nsize = 94071


    extractFieldsAndPCA = extractFieldsAndPCA(directory_data=directory_data,
              start=start,
              end=end,
              start_pca=start_pca,
              end_pca=end_pca,
              nsize=nsize)

    # Extracts data
    extractFieldsAndPCA.extractFields()

    # PCA on velocity fields
    #extractFieldsAndPCA.PCA(field_name='Velocity')

    # PCA on tracer field
    extractFieldsAndPCA.PCA(field_name='Tracer')

