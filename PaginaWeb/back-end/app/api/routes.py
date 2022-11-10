from flask import Blueprint, request, jsonify
from models.users import fetch
import numpy as np
from scipy import signal
from wavelets import Ricker
import os
# import cv2
import sys
import base64
import re
import codecs
import io
from datetime import datetime
from io import BytesIO
import json
# import skimage
# from PIL import Image
import requests
# try:
#     from StringIO import StringIO
# except ImportError:
#     from io import StringIO

from numpy import genfromtxt

blueprint = Blueprint("api", __name__)

@blueprint.route("/")
def hello():
    return "Hello World!"

# if __name__ == "__main__":
#     app.run()


def salvar_registro(name,tipo):
    data = {}
    data["name"]=name
    data["api"]=tipo
    data["data"]=str(datetime.timestamp(datetime.now()))
    
    json_file= read_logs_registro()
    print('Lido: ', json_file, file=sys.stderr)

    save_logs_registro(json_file,data)

def save_logs_registro(json_file,inf_log):
    with open('/home/rafael/Documentos/Projetos/Meus/CNN/cnnwebv2/back-end/arquivo_de_log.json', 'w') as outfile:
        print('Log atual: ', json_file, file=sys.stderr)
        json_file['logs'].append(inf_log)
        print('Log atual: ', json_file, file=sys.stderr)
        json.dump(json_file, outfile)


@blueprint.route('/logs', methods=("GET", ))
def read_logs_registro():
    with open('/home/rafael/Documentos/Projetos/Meus/CNN/cnnwebv2/back-end/arquivo_de_log.json', 'r') as outfile:
        return json.load(outfile)


# def readb64(base64_string):
#     playload = re.sub(
#         r"^data:image\/(?:gif|png|jpeg|bmp|webp)(?:;charset=utf-8)?;base64,", "", base64_string)
#     sbuf = StringIO()
#     img = base64.b64decode(playload)
#     img = Image.open(io.BytesIO(img)).convert('L')

#     img = np.array(img,dtype=np.uint8)

#     print('LOG: ',img.shape, file=sys.stderr)
    
#     return img

def rr_to_bpm(frequencies):
    for i in range(len(frequencies)):
        if frequencies[i] != 0:
            frequencies[i] = (frequencies[i]**(-1))*1000*60
    return frequencies


def moving_median(array):
    initiation_window = 0
    end_window = janela = 9  # janela definida como 9
    while initiation_window < (array.size-janela):
        window = array[initiation_window: end_window]
        median_window = np.median(window)
        if array[initiation_window] > median_window + 200:  # threshold de 200
            array[initiation_window] = int(median_window)
        elif array[initiation_window] < median_window - 200:  # threshold de 200
            array[initiation_window] = int(median_window)
        initiation_window = initiation_window + 1
        end_window = end_window + 1
    return array


def image(array):
    axis_x = np.linspace(0, len(array)-1, len(array))
    axis_y = array
    return axis_x.tolist(), axis_y.tolist()


def recebe_arquivo(text):
    array = []
    text = re.split("\r\n|\r|\n", text)
    for line in text:
        if len(line) != 0:
            array.append(int(line))
    array = np.asarray(array, dtype=np.intc)
    return array


def Filtro(array):
    if(len(array) >= 348):
        filtered = moving_median(array)
        for i in range(348, len(filtered)):
            filtered = np.delete(filtered, 348)
    return filtered


def Wavelet(array):
    frequenciesWavalet = []
    for line in array:
        frequenciesWavalet.append(int(line))
    bpm_values = rr_to_bpm(frequenciesWavalet)
    bpm_np = np.array(bpm_values)
    widths = np.arange(1, 31)
    MatrWavelet = signal.cwt(bpm_np, signal.ricker, widths)
    return MatrWavelet.tolist()


def tratamentoMatrizWavelet(Matrix):
    xtt = []
    xtt.append(Matrix)
    xg = np.array(xtt)
    xg = xg.astype('float32')/371.041
    return xg.reshape(-1, 30, 348, 1)


def VFC(data):
    frequencies = recebe_arquivo(data['conteudoArquivo'])
    Plot_Original_x_axis, Plot_Original_y_axis = image(frequencies)

    filtered = Filtro(frequencies)

    Plot_Filtrada_x_axis, Plot_Filtrada_y_axis = image(filtered)

    Matr_Wavelet = Wavelet(filtered)

    xg = tratamentoMatrizWavelet(Matr_Wavelet)

    payload = {
        "instances": xg.tolist()

    }

    r = requests.post(
        'http://localhost:8501/v1/models/vfc:predict', json=payload)
    pred = json.loads(r.content.decode('utf-8'))
    # print('LOG: '+'Request:',pred, file=sys.stderr)
    # print('LOG: '+'Request:',    model.predict(xg), file=sys.stderr)

    msg = {
        "Saudavel": "%.2f" % (pred["predictions"][0][2]*100),
        "Hipertensao": "%.2f" % (pred["predictions"][0][1]*100),
        "Arritmia": "%.2f" % (pred["predictions"][0][0]*100),
        "Plot_Original_x_axis": Plot_Original_x_axis,
        "Plot_Original_y_axis": Plot_Original_y_axis,
        "Plot_Filtrada_x_axis": Plot_Filtrada_x_axis,
        "Plot_Filtrada_y_axis": Plot_Filtrada_y_axis,
        "Plot_Wavelet": Matr_Wavelet,
        "Erro": False
    }
    return msg


# def readb64(base64_string):
#     playload = re.sub(
#         r"^data:image\/(?:gif|png|jpeg|bmp|webp)(?:;charset=utf-8)?;base64,", "", base64_string)
#     sbuf = StringIO()
#     img = base64.b64decode(playload)
#     img = Image.open(io.BytesIO(img)).convert('L')

#     img = np.array(img,dtype=np.uint8)

#     print('LOG: ',img.shape, file=sys.stderr)
    
#     return img
    
    


# def equalizar_imagem(img):
#     img = cv2.equalizeHist(img)
#     return img


# def encode(image):

#     # convert image to bytes
#     with BytesIO() as output_bytes:
#         PIL_image = Image.fromarray(skimage.img_as_ubyte(image))
#         PIL_image.save(output_bytes, 'JPEG') # Note JPG is not a vaild type here
#         bytes_data = output_bytes.getvalue()

#     # encode bytes to base64 string
#     base64_str = "data:image/jpeg;base64,"+str(base64.b64encode(bytes_data), 'utf-8')
    # return base64_str

# def decodifica_imagen_base(image_base_64):
#     img_new = readb64(image_base_64)

#     img = np.resize(img_new, (256, 256))

#     imagem_equalizada = equalizar_imagem(img)

#     imagem_equalizada_predicao = np.reshape(imagem_equalizada, (-1, 256, 256, 1))
    
#     axis_x = np.linspace(0, 255, 256)

#     normalize_histograma = np.histogram(imagem_equalizada.ravel(), bins=axis_x)
#     image_histograma = np.histogram(img.ravel(), bins=axis_x)

    # print('LOG: ',normalize_histograma, file=sys.stderr)
    # print('LOG: ',normalize_histograma, file=sys.stderr)



    

    # imagem_equalizada_base64 =  encode(equalizar_imagem(img_new))


    # payload = {
    #     "instances": imagem_equalizada_predicao.tolist()
    # }

    # r = requests.post(
    #     'http://localhost:8502/v1/models/covid19:predict', json=payload)
    # pred = json.loads(r.content.decode('utf-8'))

    # print('LOG: ',imagem_histograma, file=sys.stderr)
    # print('LOG: ',hist_x_antes, file=sys.stderr)

    # print('LOG: ',pred, file=sys.stderr)

    # response_image = {
    #         'predic':{
    #              "covid19": "%.2f" % (pred["predictions"][0][0]*100),
    #              "saudavel": "%.2f" % (pred["predictions"][0][1]*100)       
    #         },
  
    # response_image = {
    #         'predic':{
    #              "covid19": "%.2f" % (pred["predictions"][0][0]*100),
    #              "saudavel": "%.2f" % (pred["predictions"][0][1]*100)
    #             #  "outros": "%.2f" % (pred["predictions"][0][2]*100)              
    #         },
    #         'imagem':{
    #                 "imagem_histograma_antes":{
    #                     "x":image_histograma[1].tolist(),
    #                     "y":image_histograma[0].tolist()
    #                 },
    #                 "imagem_histograma":{
    #                     "x":normalize_histograma[1].tolist(),
    #                     "y":normalize_histograma[0].tolist()
    #                 },
    #                 "imagem_equalizada": str(imagem_equalizada_base64)
    #         }
            # ,
            # "result_imagem":{
                    
                                # } 
            # "covid19": "%.2f" % (pred["predictions"][0][2]*100),
            # "pneumonia": "%.2f" % (pred["predictions"][0][1]*100),
            # "saudavel": "%.2f" % (pred["predictions"][0][0]*100),]
    # }
       
    # return response_image


@blueprint.route('/api', methods=("POST", ))
def api():
    data = request.get_json(force=True)
    salvar_registro(data['name'],data['tipo'])

    print('LOG: ', data['tipo'], file=sys.stderr)

    # if(data['tipo'] == "VFC"):
    return jsonify(VFC(data))
