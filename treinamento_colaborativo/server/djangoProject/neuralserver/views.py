import pickle
import time
from multiprocessing import Queue

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

loss_dict = {}
max_loss_dict_size = 2
share_loss_dict = {}


@csrf_exempt
def receive_loss(request):
    print("Recebendo Loss: ", request.headers["model-name"])
    global loss_dict, max_loss_dict_size, share_loss_dict

    loss_dict[request.headers["model-name"]] = request.body

    # Colocamos uma copia do dicionario para cada cliente, se ele ja estiver completo
    if len(loss_dict.keys()) >= max_loss_dict_size:
        for key in loss_dict.keys():
            share_loss_dict[key] = loss_dict

        loss_dict = {}

    return HttpResponse()


def sinc_loss(request):
    print("Compartilhando Loss: ", request.headers["model-name"])

    while request.headers["model-name"] not in share_loss_dict.keys():
        time.sleep(0.1)

    # O tipo de conteudo apenas formaliza que esta sendo transmitido dados binarios arbitrarios
    return HttpResponse(pickle.dumps(share_loss_dict.pop(request.headers["model-name"], None)), headers={"content-type": "application/octet-stream"})


weight_dict = {}
max_weight_dict_size = 2
share_weight_dict = {}


@csrf_exempt
def receive_weight(request):
    print("Recebendo weight: ", request.headers["model-name"])
    global weight_dict, max_weight_dict_size, share_weight_dict

    weight_dict[request.headers["model-name"]] = request.body

    # Colocamos uma copia do dicionario para cada cliente, se ele ja estiver completo
    if len(weight_dict.keys()) >= max_weight_dict_size:
        for key in weight_dict.keys():
            share_weight_dict[key] = weight_dict

        weight_dict = {}

    return HttpResponse()


def sinc_weight(request):
    print("Compartilhando weight: ", request.headers["model-name"])

    while request.headers["model-name"] not in share_weight_dict.keys():
        time.sleep(0.1)

    # O tipo de conteudo apenas formaliza que esta sendo transmitido dados binarios arbitrarios
    return HttpResponse(pickle.dumps(share_weight_dict.pop(request.headers["model-name"], None)), headers={"content-type": "application/octet-stream"})
