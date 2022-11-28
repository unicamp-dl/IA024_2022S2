import pickle
import urllib.parse

import requests


class ServerInterface(object):
    def __init__(self, model_name, server_adress="https://patrickctrf.loca.lt/", ):
        self.server_adress = server_adress
        self.model_name = model_name

        self.clienteHttp = requests.session()

    def share_loss(self, loss_tensor):
        response_code = 0

        # codigo 200 significa que o request deu certo
        while response_code != 200:
            try:
                response = self.clienteHttp.post(
                    url=urllib.parse.urljoin(self.server_adress, "neuralserver/receive_losses"),
                    data=pickle.dumps(loss_tensor),
                    headers={"model-name": self.model_name},
                    timeout=None,
                )
                response_code = response.status_code
            except requests.exceptions.Timeout as e:
                print("Trying request share_loss again")

            print("Response status code: ", response_code)

        return response

    def receive_losses(self, ):
        response_code = 0

        # codigo 200 significa que o request deu certo
        while response_code != 200:
            try:
                response = self.clienteHttp.get(
                    url=urllib.parse.urljoin(self.server_adress, "neuralserver/sinc_losses"),
                    headers={"model-name": self.model_name},
                    timeout=None,
                )
                response_code = response.status_code
            except requests.exceptions.Timeout as e:
                print("Trying request receive_losses again")

            print("Response status code: ", response_code)

        # A resposta do servidor eh um dict serializado
        response_dict = pickle.loads(response.content)

        # Os values() do dicionario tb estao serializados. Entao desserializamos
        for key in response_dict.keys():
            response_dict[key] = pickle.loads(response_dict[key])

        return response_dict

    def share_weights(self, weights_tensor):
        response_code = 0

        # codigo 200 significa que o request deu certo
        while response_code != 200:
            try:
                response = self.clienteHttp.post(
                    url=urllib.parse.urljoin(self.server_adress, "neuralserver/receive_weights"),
                    data=pickle.dumps(weights_tensor),
                    headers={"model-name": self.model_name},
                    timeout=None,
                )
                response_code = response.status_code
            except requests.exceptions.Timeout as e:
                print("Trying request share_weights again")

            print("Response status code: ", response_code)

        return response

    def receive_weights(self, ):
        response_code = 0

        # codigo 200 significa que o request deu certo
        while response_code != 200:
            try:
                response = self.clienteHttp.get(
                    url=urllib.parse.urljoin(self.server_adress, "neuralserver/sinc_weights"),
                    headers={"model-name": self.model_name},
                    timeout=None,
                )
                response_code = response.status_code
            except requests.exceptions.Timeout as e:
                print("Trying request receive_weights again")

            print("Response status code: ", response_code)

        # A resposta do servidor eh um dict serializado
        response_dict = pickle.loads(response.content)

        # Os values() do dicionario tb estao serializados. Entao desserializamos
        for key in response_dict.keys():
            response_dict[key] = pickle.loads(response_dict[key])

        return response_dict
