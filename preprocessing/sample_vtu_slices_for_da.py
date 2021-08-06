"""
For pre-processing use this file to crop vtu images.
"""
import numpy as np
import scipy as sp
import pyvista as pv
import vtk
import sys, os
import pyvista as pv
from pyvista import examples
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':

    directory = 'G://PHD_FILES_EXTERNAL//pub/COVID_ROM_ACH_2_5_heating_P0DG//'
    snapshots_nr = 100

    for i in range(1,snapshots_nr):
        pubfile = 'Small_pub_' + str(i) + '.vtu'
        #choose extension to save file.svg, .eps, .ps, .pdf, .tex
        savepath = 'G://PHD_FILES_EXTERNAL//pub/COVID_ROM_ACH_2_5_heating_P0DG//da_slices'+'//Small_pub_' + str(i) +'.svg'
        print("processing file: ", pubfile)
        filename = os.path.join(directory, pubfile)

        mesh = pv.read(filename)
        #Set active scalar
        pv.set_plot_theme("paraview")
        mesh.set_active_scalars('PassiveTracer_CO2')
        single_slice = mesh.slice(normal=[0, 1, 0], origin = [3.75, 4.25, 1.5])
        cmap = 'Greys_r'#'jet'
        p = pv.Plotter()
        p.add_mesh(single_slice, cmap=cmap,clim=[0.0004,0.0716745]) #minmax scale here!
        p.remove_scalar_bar()

        #Camera
        p.camera_position = 'xz'
        p.camera.focal_point = (5.057, 1.165, 1.75) #x y z
        p.camera.camera_position = (-19,-207,-3.1)
        p.camera.up = (0,0,1)
        p.camera.GetWindowCenter = (3.75,4.26,1.5)
        p.camera.SetViewAngle = (0.211)
        p.camera.parralel_scale = (4.038)
        p.camera.GetParallelProjection = (0)
        p.camera.zoom(10.5)
        p.scale
        p.save_graphic(savepath)
        p.show()


    # use "free svg converter" later to convert image
    # show as pyplot image and save as png file for easy import in cv2. Careful: Resolution will be lower
        if 0:
            savepath = 'G://PHD_FILES_EXTERNAL//pub/COVID_ROM_ACH_2_5_heating_P0DG//da_slices'+'//Small_pub_' + str(i)

            fig, ax = plt.subplots()
            fig = plt.imshow(p.image)
            ax.axis("off") 
            #fig.savefig('test_da')
            plt.savefig(savepath)
