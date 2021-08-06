"""
Run this to add Generated fields back to .vtu pub file
"""

import sys, os
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import git_vtk_tools as vtktools

#%% load ps_pcae and ps_pc
if __name__ == '__main__':

    ps_pc = np.load('G:/PHD_FILES_EXTERNAL/pub/COVID_ROM_ACH_2_5_heating_P0DG/data_1_to_680/PC_tau_4_Tracer_data_1_to_680.npy')
    ps_pcae = np.load('G:/PHD_FILES_EXTERNAL/pub/COVID_ROM_ACH_2_5_heating_P0DG/data_1_to_680/PCAE_tau_4_Tracer_data_1_to_680.npy')

    #load vtu file
    start = 1
    end = 680
    path = 'G:/PHD_FILES_EXTERNAL/pub/COVID_ROM_ACH_2_5_heating_P0DG/'
    filepath = path+'/Small_pub_1.vtu'
    mesh = pv.read(filepath)
    variable = "PassiveTracer_CO2" #'Temperature'
    tracer_mesh = mesh.get_array(variable)

    #add fields to vtu
    for i in range(start, end+1):
        filename = path+'Small_pub_' + str(i) + '.vtu'
        ug = vtktools.vtu(filename)
        #ug.RemoveField('ps_pcae_full')
        #ug.RemoveField('ps_pc_full')
        ug.AddVectorField('ps_pcae_full', np.transpose(ps_pcae[i-1:i,:]))#Velocities (goes in as a 3D array)
        ug.AddVectorField('ps_pc_full', np.transpose(ps_pc[i-1:i,:]))#Velocities (goes in as a 3D array)
        ug.Write(filename)


    # test if fields are aded right
    if 0:
        path = 'G:/PHD_FILES_EXTERNAL/pub/COVID_ROM_ACH_2_5_heating_P0DG/'
        filepath = path+'/Small_pub_1.vtu'
        mesh = pv.read(filepath)
        variable = "PassiveTracer_CO2" #'Temperature'
        tracer_mesh = mesh.get_array(variable)
        plt.plot(tracer_mesh)
        variable = "ps_pcae" #'Temperature'
        tracer_mesh = mesh.get_array(variable)
        plt.plot(tracer_mesh)

