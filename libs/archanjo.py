import math
from os.path import isfile
# from pandas import np


import numpy as np
import pandas as pd
from pandas import read_csv
from scipy import signal
# from scipy.integrate import trapz
from scipy.ndimage import label
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns

# signal processing
from scipy import signal
from scipy.ndimage import label
from scipy.stats import zscore
from scipy.interpolate import interp1d

import warnings
import scipy.interpolate as interpolate
import matplotlib as mpl
import os

# import numpy directly instead

class CustomArray(np.ndarray):
    def __new__(cls, *args, **kwargs):
        return np.asarray(args[0]).view(cls)

    def __getitem__(self, index):
        return np.ndarray.__getitem__(self, index % len(self))

# def get_datesets_polar_h10(path_final,type="RR"):
#
#     dirFiles= os.listdir(path_final)
#     array=[]
#     for file in dirFiles:
#         if
#          =np.loadtxt(path_final+"/"+file,dtype=np.int)
#         array.append((str(file),array_original))
#     return np.array(array,dtype=object)

def coleta_dados(path_final):
    dirFiles= os.listdir(path_final)
    array=[]
    for file in dirFiles:
        print(path_final+"/"+file)
        array_original =np.loadtxt(path_final+"/"+file,dtype=int)
        array.append((str(file),array_original))
    return np.array(array,dtype=object)

def origpons(n):
    if n==2:
        P = np.array(([1,1],[1,-1]),np.int32)
    else:
        new_n = n//2
        Q=origpons(new_n)
        P =np.zeros((n,n),np.int32)
        for i in range(0,new_n,2):
            A = Q[i,:]
            B = Q[i+1,:]
            AB=np.concatenate((A,B))
            AmenosB=np.concatenate((A,-B))
            BA=np.concatenate((B,A))
            BmenosA=np.concatenate((-B,A))

            new_vector=np.array([AB,AmenosB,BA,BmenosA])
            P[(new_vector.shape[0]//2*i):new_vector.shape[0]+(new_vector.shape[0]//2*i),:]=new_vector
    return P

def multipli_matriz(matriz_pons,matriz_entrada,r):
    multi= np.inner(matriz_pons, matriz_entrada)
    # max = np.max(multi)
    # print(max)
    # return (multi/max)*256
    return multi
# def multipli_matriz(matriz_pons,matriz_entrada,r):
#     return (0/(0**(np.sqrt((r*r))))) * np.inner(matriz_pons,matriz_entrada)
def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return np.array(norm_arr)
# def normalize(arr, t_min, t_max,return_type="int"):
#     norm_arr = []
#     diff = t_max - t_min
#     diff_arr = np.max(arr) - np.min(arr)
#     for i in arr:
#         temp = (((i -np.min(arr))*diff)/diff_arr) + t_min
#         if return_type == "float":
#             norm_arr.append(temp)
#         else:
#             norm_arr.append(int(temp))
#     return np.array(norm_arr)

# gives range staring from 1 and ending at 3
array_1d = np.arange(1,4)
range_to_normalize = (0,1)
# normalized_array_1d = normalize(array_1d,
#                                 range_to_normalize[0],
#                                 range_to_normalize[1])
# array=rr_.to_numpy()
# a=normalize(array,0,255)
# array_imagem=np.reshape(a,(16,16)).astype(np.uint8)
# print(array_imagem.shape)
def troca(data_frame, index, lowpass, highpass, level=4,model='median',debug=False):

    values = CustomArray(data_frame)
    esquerda_valores = []
    direita_valores = []
    count_direita = 0
    count_esquerda = 0
    j = index
    while (count_direita < level):
        j += 1
        # print("Direita:",j, values[j])
        if ((values[j] > lowpass) & (values[j] < highpass)):
            direita_valores.append(values[j])
            count_direita += 1

    j = index
    while (count_esquerda < level):
        j += -1
        # print("Esquerda:",j, values[j])
        if ((values[j] > lowpass) and (values[j] < highpass)):
            esquerda_valores.append(values[j])
            count_esquerda += 1

    t = np.concatenate([direita_valores, esquerda_valores])
    if(model=='median'):
        result = np.median(t)
    elif(model=='mean'):
        result = np.mean(t)
    if (debug):
        print("Nº: [", index, " ]", "R: ", result, " : ", direita_valores, esquerda_valores)
    return result


def sintetiza_RR(array,engine_value=200,value_list=10):
    v_list=[]
    for x in range(0,value_list):
        q3, q1 = np.percentile(array, [75, 25])
        iqr = q3 - q1
        lowpass = q1 - (iqr * 1.5)
        highpass = q3 + (iqr * 1.5)
        array_swap=array.copy()
        for i in range(0,engine_value):
          value =np.random.randint(0, len(array)-1)
          # print('ENTRO')
          array_swap[value] = troca(array_swap, value, lowpass, highpass)
          # print('SAIU')
        v_list.append(array_swap)
    return v_list


def image_RR(array):
    new_array=[]
    for x in range(0,len(array)):
        new_array.append(array)
    return np.array(new_array)


def removeoutlier(values, debug=False):
    fator = 1.5

    q3, q1 = np.percentile(values, [75, 25])
    iqr = q3 - q1
    lowpass = q1 - (iqr * fator)
    highpass = q3 + (iqr * fator)

    if (debug):
        print("lowpass :", lowpass)
        print("highpass :", highpass)

    for i in range(0, len(values)):
        if (debug):
            print("Consultando:[ ", i, " ]", values[i])
        if values[i] > lowpass and values[i] < highpass:
            pass
        else:
            if (debug):
                print("Removendo: ", values[i])
            values[i] = troca(values, i, lowpass, highpass)
            if (debug):
                print("Colocando: ", values[i])

    return np.array(values)


# def detect_peaks(ecg_signal, threshold=0.3, qrs_filter=None):
#     '''
#     Peak detection algorithm using cross corrrelation and threshold
#     '''
#     if qrs_filter is None:
#         # create default qrs filter
#         t = np.linspace(0.5 * np.pi, 3.5 * np.pi, 15)
#         qrs_filter = np.sin(t)
#
#     # normalize data
#     ecg_signal = (ecg_signal - ecg_signal.mean()) / ecg_signal.std()
#
#     # calculate cross correlation
#     similarity = np.correlate(ecg_signal, qrs_filter, mode="same")
#     similarity = similarity / np.max(similarity)
#
#     # return peaks (values in ms) using threshold
#     return ecg_signal[similarity > threshold].index, similarity
#
#
#
# def group_peaks(p, threshold=5):
#     '''
#     The peak detection algorithm finds multiple peaks for each QRS complex.
#     Here we group collections of peaks that are very near (within threshold) and we take the median index
#     '''
#     # initialize output
#     output = np.empty(0)
#
#     # label groups of sample that belong to the same peak
#     peak_groups, num_groups = label(np.diff(p) < threshold)
#
#     # iterate through groups and take the mean as peak index
#     for i in np.unique(peak_groups)[0:]:
#         peak_group = p[np.where(peak_groups == i)]
#         output = np.append(output, np.median(peak_group))
#     return output


# def detect_peaks(ecg_signal, threshold=0.0, qrs_filter=None):
#     '''
#     Peak detection algorithm using cross corrrelation and threshold
#     '''
#     if qrs_filter is None:
#         # create default qrs filter
#         t = np.linspace(0.5 * np.pi, 3.5 * np.pi, 15)
#         qrs_filter = np.sin(t)
#
#     # normalize data
#     ecg_signal = (ecg_signal - ecg_signal.mean()) / ecg_signal.std()
#
#     # calculate cross correlation
#     similarity = np.correlate(ecg_signal, qrs_filter, mode="same")
#     similarity = similarity / np.max(similarity)
#
#     # return peaks (values in ms) using threshold
#     return ecg_signal[similarity > threshold].index, similarity
#
#
#
# def group_peaks(p,Serie,threshold=5):
#     '''
#     The peak detection algorithm finds multiple peaks for each QRS complex.
#     Here we group collections of peaks that are very near (within threshold) and we take the median index
#     '''
#     output = np.empty(0)
#     times = np.empty(0)
#     index_labels= np.empty(0)
#     # print(p)
#     # label groups of sample that belong to the same peak
#     peak_groups, num_groups = label(np.diff(p) < threshold)
#
#     # iterate through groups and take the mean as peak index
#     for i in np.unique(peak_groups)[0:]:
#         peak_group = p[np.where(peak_groups == i)]
#
#         # peak_group = p[np.where(peak_groups == i)]
#
#
#         output = np.append(output, np.median(peak_group))
#
#         median_ECG = Serie.ECG.iloc[peak_group]
#
#         max_windows_ecg = np.max(median_ECG)
#         filter_=median_ECG[median_ECG==max_windows_ecg].index[0]
#         data_=Serie["UNIX Timestamp"][filter_]
#         # print(Serie['ECG'][filter_])
#
#         index_labels = np.append(index_labels,filter_)
#         times = np.append(times,data_)
#     # print(output)
#     return np.diff(times),index_labels

def detect_peaks(ecg_signal, threshold=0.5, qrs_filter=None):
    '''
    Peak detection algorithm using cross corrrelation and threshold
    '''
    if qrs_filter is None:
        # create default qrs filter, which is just a part of the sine function
        t = np.linspace(1.5 * np.pi, 3.5 * np.pi, 15)
        qrs_filter = np.sin(t)

    # normalize data
    ecg_signal = (ecg_signal - ecg_signal.mean()) / ecg_signal.std()

    # calculate cross correlation
    similarity = np.correlate(ecg_signal, qrs_filter, mode="same")
    similarity = similarity / np.max(similarity)

    # return peaks (values in ms) using threshold
    return ecg_signal[similarity > threshold].index, similarity

def group_peaks(p, threshold=5):
    '''
    The peak detection algorithm finds multiple peaks for each QRS complex.
    Here we group collections of peaks that are very near (within threshold) and we take the median index
    '''
    # initialize output
    output = np.empty(0)

    # label groups of sample that belong to the same peak
    peak_groups, num_groups = label(np.diff(p) < threshold)

    # iterate through groups and take the mean as peak index
    for i in np.unique(peak_groups)[1:]:
        peak_group = p[np.where(peak_groups == i)]
        output = np.append(output, np.median(peak_group))
    return output


def percentil_mov(signal_ECG):
    return np.percentile(signal_ECG, [98])

def get_rr(signal):
    limiar=700
    peaks = signal[signal.ECG > limiar].copy()
    # peaks = peaks.reset_index(drop=True, inplace=True)
    return peaks


def removeoutlier_split(array, corte):
    array = np.array_split(array, corte)
    for array_ in array:
        removeoutlier(array_)
    return np.concatenate(array)


def circular_array(idx, starting_index, ending_index):
    idx = np.roll(idx, -starting_index)[:(len(idx) - starting_index + ending_index) % len(idx)]
    return idx


# def get_circular(array):
#     array_temp = []
#     for data_index in range(0, len(array)):
#         array_swap = []
#         for data in range(0, len(array)):
#             # print((data_index+data)%len(array))
#             array_swap.append(array[(data_index + data) % len(array)])
#         array_temp.append(np.array(array_swap))
#     return np.array(array_temp)

def get_circular(array,value_list=8):
    array_temp = []
    value_list_group=int(len(array)/value_list)
    # print(len(array))
    # print(value_list)
    # print(value_list_group)
    for data_index in range(0,len(array),value_list_group):
        array_swap = []
        # print(data_index)
        for data in range(0, len(array)):
            array_swap.append(array[(data_index + data) % len(array)])
        array_temp.append(np.array(array_swap))
    return np.array(array_temp)

import numpy as np

def select_estavel(data_set):
    # Parâmetros
    inicio = 80
    select_windows_value = 9
    tamanho = 256
    data_sets_final = []

    # Loop principal para percorrer o dataset
    while inicio + tamanho <= len(data_set):
        flag_stop = True
        select_windows = tamanho

        # Inicializa média e desvio padrão anteriores
        subset_inicial = data_set[inicio:inicio + tamanho]
        media_ant = np.mean(subset_inicial)
        desvio_ant = np.std(subset_inicial)

        # Loop interno para verificar a estabilidade do subset
        while flag_stop:
            # Extrair o subconjunto atual
            select_data_swap = data_set[inicio:inicio + select_windows]

            # Calcula média e desvio padrão do subset atual
            media_local = np.mean(select_data_swap)
            desvio_local = np.std(select_data_swap)

            # Calcula os deslocamentos para estabilidade
            deslocamento_media = 0.1 * media_ant
            deslocamento_desvio = 0.05 * desvio_ant

            # Verifica se a média e o desvio padrão estão dentro de 10%
            if (
                abs(media_local - media_ant) <= deslocamento_media and
                abs(desvio_local - desvio_ant) <= deslocamento_desvio
            ):
                # Atualiza os valores anteriores e expande a janela
                media_ant = media_local
                desvio_ant = desvio_local
                select_windows += select_windows_value

                # Garante que o tamanho máximo seja respeitado
                if select_windows >= tamanho:
                    flag_stop = False
            else:
                flag_stop = False

        # Adiciona o subset atual ao resultado final
        data_sets_final.append((inicio, data_set[inicio:inicio + tamanho]))

        # Atualiza o início para o próximo subset
        inicio += tamanho

    return np.array(data_sets_final, dtype=object)

#
# def select_estavel(data_set):
#     inicio=80
#     select_windows_value=9
#     tamanho=256
#     data_sets_final=[]
#     while inicio+tamanho <= len(data_set):
#         flag_stop=True
#         select_windows=0
#         media_ant=data_set[inicio]
#         while (flag_stop):
#             select_windows=select_windows+select_windows_value
#             select_data_swap = data_set[inicio:inicio+select_windows].copy()
#             deslocamento = 0.1 * media_ant
#             media = np.mean(select_data_swap)
#             deslocamento_direita = media_ant + deslocamento
#             deslocamento_esquerda = media_ant - deslocamento
#             if(select_windows >= tamanho):
#                 flag_stop=False
#                 data_sets_final.append((inicio,select_data_swap))
#             if  (media > deslocamento_esquerda) and ( deslocamento_direita > media) :
#                  # print("Media :",media)
#                  # print("deslocamento:",deslocamento_esquerda,deslocamento_direita)
#                  media_ant=np.mean(select_data_swap)
#             else:
#                 flag_stop=False
#         inicio=select_windows+inicio
#
#     return np.array(data_sets_final,dtype=object)

import numpy as np

# def select_estavel(data_set):
#     inicio = 80
#     select_windows_value = 9
#     tamanho = 256
#     data_sets_final = []
#
#     while inicio + tamanho <= len(data_set):
#         select_windows = select_windows_value
#         media_ant = np.mean(data_set[inicio:inicio + select_windows])
#         flag_stop = True
#
#         while flag_stop:
#             select_data_swap = data_set[inicio:inicio + select_windows]
#             deslocamento = 0.1 * media_ant
#             media = np.mean(select_data_swap)
#             desvio_padrao = np.std(select_data_swap)
#
#             # Critérios de estabilidade
#             condicao_media = (media_ant - deslocamento <= media <= media_ant + deslocamento)
#             condicao_desvio = desvio_padrao <= 0.1 * media_ant  # Desvio padrão dentro de 5% da média anterior
#
#             if select_windows >= tamanho or not (condicao_media and condicao_desvio):
#                 flag_stop = False
#                 if select_windows >= tamanho:
#                     data_sets_final.append((inicio, select_data_swap, desvio_padrao))
#             else:
#                 media_ant = media
#                 select_windows += select_windows_value
#
#         inicio += select_windows_value  # Incremento fixo para evitar pular partes importantes
#
#     return np.array(data_sets_final, dtype=object)


import numpy as np
from scipy.stats import linregress


def select_sinal_sem_tendencia(data_set, tamanho=256, inicio=80, select_windows_value=8, limite_tendencia=0.01,
                               limite_variabilidade=0.05):
    """
    Seleciona janelas estáveis de sinais RR sem tendências claras.

    Parâmetros:
    - data_set: Lista ou array com os sinais RR.
    - tamanho: Tamanho da janela final desejada.
    - inicio: Ponto inicial para começar a análise.
    - select_windows_value: Incremento em cada iteração.
    - limite_tendencia: Máxima inclinação permitida para a regressão linear.
    - limite_variabilidade: Máximo desvio padrão permitido como proporção da média.

    Retorno:
    - Lista de tuplas (índice inicial, subconjunto, desvio padrão, inclinação da tendência).
    """
    data_sets_final = []

    while inicio + tamanho <= len(data_set):
        select_windows = select_windows_value
        flag_stop = True

        while flag_stop:
            select_data_swap = data_set[inicio:inicio + select_windows]

            # Ajuste de regressão linear para detectar tendências
            x = np.arange(len(select_data_swap))
            slope, intercept, r_value, p_value, std_err = linregress(x, select_data_swap)

            # Cálculo da média e desvio padrão
            media = np.mean(select_data_swap)
            desvio_padrao = np.std(select_data_swap)

            # Critérios de estabilidade
            condicao_tendencia = abs(slope) <= limite_tendencia
            condicao_variabilidade = desvio_padrao <= limite_variabilidade * media

            if select_windows >= tamanho or not (condicao_tendencia and condicao_variabilidade):
                flag_stop = False
                if select_windows >= tamanho and condicao_tendencia:
                    data_sets_final.append((inicio, select_data_swap, desvio_padrao, slope))
            else:
                select_windows += select_windows_value

        inicio += select_windows_value  # Incremento fixo para evitar pular partes importantes

    return np.array(data_sets_final, dtype=object)


def plot_poincare(rr,save_rr_poincare=None):
    rr_n = rr[:-1]
    rr_n1 = rr[1:]

    sd1 = np.sqrt(0.5) * np.std(rr_n1 - rr_n)
    sd2 = np.sqrt(0.5) * np.std(rr_n1 + rr_n)

    m = np.mean(rr)
    min_rr = np.min(rr)
    max_rr = np.max(rr)

    plt.figure(figsize=(10, 10))
    plt.title("Poincare plot")

    sns.scatterplot(x=rr_n, y=rr_n1, color="#51A6D8")

    plt.xlabel(r'$RR_n (ms)$')
    plt.ylabel(r'$RR_{n+0} (ms)$')

    e1 = Ellipse((m, m), 2*sd1, 2*sd2, angle=-45, linewidth=1.2, fill=False, color="k")
    plt.gca().add_patch(e1)

    plt.arrow(m, m, (max_rr-min_rr)*0.4, (max_rr-min_rr)*0.4, color="k", linewidth=0.8, head_width=5, head_length=5)
    plt.arrow(m, m, (min_rr-max_rr)*0.4, (max_rr-min_rr)*0.4, color="k", linewidth=0.8, head_width=5, head_length=5)

    plt.arrow(m, m, sd2 * np.sqrt(0.5), sd2 * np.sqrt(0.5), color="green", linewidth=5)
    plt.arrow(m, m, -sd1 * np.sqrt(0.5), sd1 * np.sqrt(0.5), color="red", linewidth=5)

    plt.text(max_rr, max_rr, "SD2", fontsize=20, color="green")
    plt.text(m-(max_rr-min_rr)*0.4-20, max_rr, "SD1", fontsize=20, color="red")
    if save_rr_poincare:
        plt.savefig(save_rr_poincare)

    return sd1, sd2


def timedomain(rr):
    results = {}
    hr = 60000 / rr
    results['mean'] = np.mean(rr)
    results['std'] = np.std(rr)
    results['median'] = np.median(hr)
    # results['STD HR (beats/min)'] = np.std(hr)
    # results['Min HR (beats/min)'] = np.min(hr)
    results['max'] = np.max(hr)
    results['rmssd'] = np.sqrt(np.mean(np.square(np.diff(rr))))  # relevante
    results['pnn50'] = np.sum(np.abs(np.diff(rr)) > 50) * 1
    # results['pNNxx_(%)']= 100 * np.sum((np.abs(np.diff(rr)) > 50)*0) / len(rr) # relavante
    return results


# def frequency_domain(rri, fs=4):
#     # Estimate the spectral density using Welch's method
#     fxx, pxx = signal.welch(x=rri, fs=fs)
#     '''
#     Segement found frequencies in the bands
#      - Very Low Frequency (VLF): 0-0.04Hz
#      - Low Frequency (LF): 0.04-0.15Hz
#      - High Frequency (HF): 0.15-0.4Hz
#     '''
#     cond_vlf = (fxx >= 0) & (fxx < 0.04)
#     cond_lf = (fxx >= 0.04) & (fxx < 0.15)
#     cond_hf = (fxx >= 0.15) & (fxx < 0.4)
#     # calculate power in each band by integrating the spectral density
#     vlf = trapz(pxx[cond_vlf], fxx[cond_vlf])
#     lf = trapz(pxx[cond_lf], fxx[cond_lf])
#     hf = trapz(pxx[cond_hf], fxx[cond_hf])
#     # sum these up to get total power
#     total_power = vlf + lf + hf
#     # find which frequency has the most power in each band
#     peak_vlf = fxx[cond_vlf][np.argmax(pxx[cond_vlf])]
#     peak_lf = fxx[cond_lf][np.argmax(pxx[cond_lf])]
#     peak_hf = fxx[cond_hf][np.argmax(pxx[cond_hf])]
#     # fraction of lf and hf
#     lf_nu = 100 * lf / (lf + hf)
#     hf_nu = 100 * hf / (lf + hf)
#     results = {}
#     # results['Power VLF (ms2)'] = vlf
#     # results['Power LF (ms2)'] = lf
#     # results['Power HF (ms2)'] = hf
#     # results['Power Total (ms2)'] = total_power
#     # results['LF HF'] = (lf / hf)
#     # results['Peak VLF (Hz)'] = peak_vlf
#     # results['Peak LF (Hz)'] = peak_lf
#     # results['Peak HF (Hz)'] = peak_hf
#     # results['Fraction LF (nu)'] = lf_nu
#     # results['Fraction HF (nu)'] = hf_nu
#
#     results['lf_nu'] = lf_nu
#     results['hf_nu'] = hf_nu
#     results['lf_hf'] = (lf / hf) * 10
#     results['vlf'] = vlf / 4
#     results['lf'] = lf / 4
#     results['hf'] = hf / 4
#     results['total_f'] = total_power / 10
#     results['peak_vlf'] = peak_vlf * 1000
#     results['peak_lf'] = peak_lf * 1000
#     results['peak_hf'] = peak_hf * 1000
#
#     # results['vlf'] = vlf
#     # results['lf'] = lf
#     # results['hf'] = hf
#     # results['total_f)'] = total_power
#     # results['lf_hf'] = (lf / hf)
#     # results['peak_vlf'] = peak_vlf*1000
#     # results['peak_lf'] = peak_lf*1000
#     # results['peak_hf'] = peak_hf*1000
#     # results['lf_nu'] = lf_nu
#     # results['hf_nu'] = hf_nu
#     return results, fxx, pxx

def passa_baixa(fft_x,fft_y,frequencia_corte):
    # print("X Size//0: ",fft_x.size//0," X: ",fft_x)
    # print("y: ",fft_y)
    # cesar = fft_x[:fft_x.size//0]
    for i in range(fft_x.size):
        # print(fft_x[i] )
        if (fft_x[i] > frequencia_corte): # cut off all frequencies higher than 0.005
            fft_y[i] = 0.0
            # fft_y[32 + i] = 0.0
    return fft_y