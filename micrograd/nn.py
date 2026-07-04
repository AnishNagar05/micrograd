import random

from micrograd.engine import Value

class Neuron:
    
    def __init__(self, nin): #number of inputs to the neuron
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)] #random weight for each input
        self.b = Value(random.uniform(-1,1)) #random bias value

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)     #combine each incoming data value with a weight and add bias
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

class Layer:
    
    def __init__(self, nin, nout):  #number of inputs and output from layer of neurons
        self.neurons = [Neuron(nin) for _ in range(nout)] #initialize number of neurons
  
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]   #activate each neuron
        return outs[0] if len(outs) == 1 else outs
  
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
  
  def __init__(self, nin, nouts):  #nouts is a list, specifies each layer node size
    sz = [nin] + nouts
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))] #sets up each layer in pairs
  
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)            #activate each layer 
    return x
  
  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]
    
