"""
Analytics: This file is run fourth and reconstructs latent space
The commented out part will save your reconstructed fields from the AE and the truncated PCs
"""

import pandas as pd
import numpy as np
import tensorflow.keras as tf
import matplotlib.pyplot as plt
from tensorflow.keras import backend


if __name__ == '__main__':
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
    directory_data = 'G:/PHD_FILES_EXTERNAL/pub/COVID_ROM_ACH_2_5_heating_P0DG/'
    field_name = 'Tracer'
    observationPeriod = 'data_1_to_680'


    start = 1
    end = 680


    npcs = 1000
    typeArch = ''
    latent_dim = 8
    epoch = 8001
    data = np.load(directory_data +  '/' + field_name + '_data_' + str(start) + '_to_' + str(end) + '.npy')
    modelPcs = np.load(directory_data +  '/' + 'pcs_' + field_name + '_'+ observationPeriod + '.npy')
    modelEofs = np.load(directory_data +  '/' + 'eofs_' + field_name + '_' + observationPeriod + '.npy')
    stdmodel = np.load(directory_data +  '/' + 'std_' + field_name +  '_' + observationPeriod + '.npy')
    meanmodel = np.load(directory_data + '/' + 'mean_' + field_name +  '_' + observationPeriod + '.npy')


    modelPcs = modelPcs[:, :npcs]
    modelPcs = modelPcs[:, :]
    xmin = np.min(modelPcs)
    xmax = np.max(modelPcs)
    pcscaled = scalerThetis(modelPcs, xmin, xmax, -1, 1)


    #latent space
    index = range(32000) 
    df = pd.DataFrame(index=index)
    j = 0
    taus = [4]
    epoch = 8001
    epoch = epoch -1
    for tau in taus:
        generator_enc = tf.models.load_model(
            directory_data  + '/' + 'AAE_MV_generator_encoder_Full_'+ GANorWGAN + '_' + field_name + '_' + str(
                tau) + '_' + str(epoch))
        generator_dec = tf.models.load_model(
            directory_data  + '/' + 'AAE_MV_generator_decoder_Full_'+GANorWGAN + '_' + field_name + '_' + str(
                tau) + '_' + str(epoch))
        columnname = ['$LS_{' + str(tau)+'}$']
        df_temp = pd.DataFrame(np.reshape(generator_enc.predict(pcscaled), -1), columns=columnname) #violin plot, histogram
        df = pd.concat([df, df_temp], axis=1)
        print(tau)
        j+=1

    #df.hist()
    ax = df.hist()
    fig = ax.get_figure()
    fig.savefig('G:/PHD_FILES_EXTERNAL/pub/COVID_ROM_ACH_2_5_heating_P0DG/')
