"""
Analytics: This file is run fifth, this will save your reconstructed fields from the AE and the truncated PCs

"""

import numpy as np
import tensorflow.keras as tf
import matplotlib.pyplot as plt
from tensorflow.keras import backend
import time

if __name__ == '__main__':
    da_states = 0 #0 preproduces standard results, 1 for experiments with augmented state vector

    plt.rcParams.update({'font.size': 18})
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rc('axes', labelsize=18)
    def scalerThetis(x, xmin, xmax, min, max):
        scale = (max - min)/(xmax - xmin)
        xScaled = scale*x + min - xmin*scale
        return xScaled
    def inverseScalerThetis(xscaled, xmin, xmax, min, max):
        scale = (max - min) / (xmax - xmin)
        xInv = (xscaled/scale) - (min/scale) + xmin
        return xInv
    directory_data = 'G:/PHD_FILES_EXTERNAL/pub/COVID_ROM_ACH_2_5_heating_P0DG/'#'/Users/cequilod/sLSBU_Simulation/'
    field_name = 'Tracer'
    observationPeriod = 'data_1_to_680'

    start = 1
    end = 680
    npcs = 1000
    typeArch = ''
    latent_dim = 8
    epoch = 8000
    data = np.load(directory_data  + '/' + field_name + '_data_' + str(start) + '_to_' + str(end) + '.npy')
    modelPcs = np.load(directory_data  + '/' + 'pcs_' + field_name + '_' + observationPeriod + '.npy')
    modelEofs = np.load(directory_data  + '/' + 'eofs_' + field_name + '_' + observationPeriod + '.npy')
    stdmodel = np.load(directory_data  + '/' + 'std_' + field_name +  '_' + observationPeriod + '.npy')
    meanmodel = np.load(directory_data  + '/' + 'mean_' + field_name +  '_' + observationPeriod + '.npy')
    start_time = time.time()

    modelPcs = modelPcs[:, :npcs]
    modelPcs = modelPcs[:, :]
    xmin = np.min(modelPcs)
    xmax = np.max(modelPcs)
    pcscaled = scalerThetis(modelPcs, xmin, xmax, -1, 1)
    latentSpaceDimensions = [4, 8, 16, 32]
    taus = [4] 
    if da_states == 1:
        pcscaled = pcscaled_xda #from reconstruct_latent_space.py line 71

    for tau in taus:
        if da_states == 1 :
            da_state = da_state_images()
            pcscaled_aug = da_state_augment(da_state,modelPcs,pca=1)
            npcs = 40 # change to e.g. 70 if state vector is augmented

        #PCAE no correction
        generator_enc = tf.models.load_model(
            directory_data +   'AAE_MV_generator_encoder_Full_'+ GANorWGAN  + '_' + field_name + '_' + str(
                tau) + '_' + str(epoch))
        generator_dec = tf.models.load_model(
            directory_data +   'AAE_MV_generator_decoder_Full_'+ GANorWGAN  + '_' + field_name + '_' + str(
                tau) + '_' + str(epoch))
        pcae = generator_dec.predict(generator_enc.predict(pcscaled))#this reconstructs the original PCs
        ps_pcae = inverseScalerThetis(pcae, xmin, xmax, -1, 1)

        if da_states == 1:
            ps_pcae = np.matmul(ps_pcae[:,:40], modelEofs[:npcs, :]) * stdmodel + meanmodel# reconstructs z in the pa
        else: 
            ps_pcae = np.matmul(ps_pcae, modelEofs[:npcs, :]) * stdmodel + meanmodel# reconstructs z in the paper


        ps_pcae = np.squeeze(ps_pcae)
        np.save(directory_data + observationPeriod + '/' + 'PCAE_tau_' + str(tau) + '_' + field_name + '_' + observationPeriod + '.npy', ps_pcae)
        del ps_pcae 

        #Truncated PCs
        ps_pc = modelPcs 
        ps_pc = np.matmul(ps_pc[:, :tau], modelEofs[:tau, :]) * stdmodel + meanmodel
        np.save(directory_data + observationPeriod + '/' + 'PC_tau_' + str(tau) + '_' + field_name + '_' + observationPeriod + '.npy', ps_pc)
        print('Done: ' + str(tau))
        del ps_pc
        print(time.time() - start_time)

