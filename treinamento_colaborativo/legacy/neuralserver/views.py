import pickle
from multiprocessing import Queue

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

loss_dict = {}
max_loss_dict_size = 2

# A fila permite que os clientes esperem a sincronizacao sem busy wait
loss_queue = Queue()


@csrf_exempt
def receive_loss(request):
    print("Recebendo Loss")
    global loss_dict, max_loss_dict_size, loss_queue

    loss_dict[request.headers["model-name"]] = request.body

    # Colocamos uma copia do dicionario para cada cliente, se ele ja estiver completo
    if len(loss_dict.keys()) >= max_loss_dict_size:
        for _ in range(max_loss_dict_size):
            loss_queue.put(loss_dict)

        loss_dict = {}

    return HttpResponse()


def sinc_loss(request):
    print("Compartilhando Loss")
    # O tipo de conteudo apenas formaliza que esta sendo transmitido dados binarios arbitrarios
    return HttpResponse(pickle.dumps(loss_queue.get()), headers={"content-type": "application/octet-stream"})


weight_dict = {}
max_weight_dict_size = 2

# A fila permite que os clientes esperem a sincronizacao sem busy wait
weight_queue = Queue()


@csrf_exempt
def receive_weight(request):
    global weight_dict, max_weight_dict_size, weight_queue

    weight_dict[request.headers["model-name"]] = request.body

    # Colocamos uma copia do dicionario para cada cliente, se ele ja estiver completo
    if len(weight_dict.keys()) >= max_weight_dict_size:
        for _ in range(max_weight_dict_size):
            weight_queue.put(weight_dict)

        weight_dict = {}

    return HttpResponse()


def sinc_weight(request):
    # O tipo de conteudo apenas formaliza que esta sendo transmitido dados binarios arbitrarios
    return HttpResponse(pickle.dumps(weight_queue.get()), headers={"content-type": "application/octet-stream"})
