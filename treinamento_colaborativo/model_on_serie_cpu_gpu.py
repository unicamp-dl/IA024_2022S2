import torch
import torch.nn as nn
from torch.autograd import Variable

dev_gpu = "cuda:0"
dev_cpu = "cpu"


class Net_cpu(nn.Module):
    def __init__(self):
        super(Net_cpu, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(10, 10),
            nn.Linear(10, 10),
            nn.Linear(10, 10),
            nn.Linear(10, 10),
        ])
        self.output = []
        self.input = []

    def forward(self, x):
        for layer in self.layers:
            # detach from previous history

            x = Variable(x.data, requires_grad=True)
            self.input.append(x)

            # compute output
            x = layer(x)

            # add to list of outputs
            self.output.append(x)

        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(10, 10),
            nn.Linear(10, 10),
            nn.Linear(10, 10),
            nn.Linear(10, 10),
        ])
        self.output = []
        self.input = []

    def forward(self, x):
        for layer in self.layers:
            # detach from previous history

            x = Variable(x.data, requires_grad=True)
            self.input.append(x)

            # compute output
            x = layer(x)

            # add to list of outputs
            self.output.append(x)

        return x


class NetTotal(nn.Module):
    def __init__(self):
        super(NetTotal, self).__init__()
        self.net_cpu = Net_cpu()
        self.net = Net()
        self.output = []
        self.input = []

    def forward(self, x):
        x = x.to(dev_gpu)
        x = self.net.to(dev_gpu)(x)

        x = x.to(dev_cpu)
        x = self.net_cpu(x)

        self.output = self.net.output + self.net_cpu.output
        self.input = self.net.input + self.net_cpu.input

        return x

    def backward(self, g):
        # reversed reverte a lista
        for i, output in reversed(list(enumerate(self.output))):
            # if i == (len(self.output) - 1) // 2:
            if i + 1 == 4: #flipar o device
                self.input[i + 1].grad.data = self.input[i + 1].grad.data.to(dev_gpu)
                self.output[i] = self.output[i].to(dev_gpu)

            #
            if i == (len(self.output) - 1):
                # for last node, use g
                self.output[i].backward(g)
            else:
                print(self.input[i + 1].grad.data.device)
                self.output[i].backward(self.input[i + 1].grad.data)
                print(f"after backward {self.input[i + 1].grad.data.sum()}")

            print(i)


model = NetTotal()

inp = Variable(torch.randn(4, 10))

output = model.forward(inp)

gradients = torch.randn(*output.size())

model.backward(gradients)
