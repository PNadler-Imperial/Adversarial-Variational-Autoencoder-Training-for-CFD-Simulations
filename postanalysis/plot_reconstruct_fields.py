"""
run sixth, this plots reconstructed fields
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



if __name__ == '__main__':
    directory_data = 'G:/PHD_FILES_EXTERNAL/pub/COVID_ROM_ACH_2_5_heating_P0DG/'
    field_name = 'Tracer'
    observationPeriod = 'data_1_to_680'
    observationPeriod1 = 'data_1_to_680'
    observationPeriod = 'data_1_to_41'
    observationPeriod1 = 'data_1_to_41'
    start = 1
    end = 680
    npcs = 1000
    typeArch = ''
    epoch = 8000 #gives  4 subplots , velocity x,y,z
    field_data = np.load(directory_data + '/' + field_name + '_data_' + str(start) + '_to_' + str(end) + '.npy')
    #dimfield = field_data.shape[1]
    dimfield = 1
    taus = [4]
    Metrics = ['MAE']
    PCorAEs = ['PCAE','PC']
    j=1
    legendnames=[]
    for metric in Metrics:
        for i in range(4):#for i in range(3):
            plt.subplot(2,2,j)
            for tau in taus:
                for PCorAE in PCorAEs:
                    p = np.load(directory_data + observationPeriod + '/' + PCorAE + '_tau_' + str(tau) +
                            '_' + field_name + '_' + observationPeriod + '.npy')
                    data_temp = field_data[ :, i]  
                    p_temp = p[:, dimfield * i:dimfield * (i + 1)]
                    misfit = data_temp - p_temp
                    if metric == Metrics[0]:
                        metric_temp = np.mean(np.abs(misfit), 1)
                    plt.plot(metric_temp)
                    legendnames.append('$' + PCorAE + '_' + str(tau) + '$')
            plt.legend(legendnames)
            j+=1
