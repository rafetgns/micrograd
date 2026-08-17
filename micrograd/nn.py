import random
from micrograd.engine import Value

class Module:
    """pytorch like inheritance of abstract base class providing the shared functionality across child classes"""

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []
        #fallback for parameters

class Neuron(Module):
    """single neuron containing #n_in weights and 1 bias"""

    def __init__(self, n_in, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(n_in)]
        self.b = Value(random.uniform(-1,1))
        self.nonlin = nonlin
    def __call__(self, x):
        values = zip(self.w, x)
        #tuples of input values and weights
        
        act = sum((value*weight for (value, weight) in values), self.b)

        act = act.relu() if self.nonlin else act
        #apply nonlinearity if flag is true

        return act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}_Neuron({len(self.w)})"

class Layer(Module):
    """layer containing n_out neurons, each with n_in weights"""

    def __init__(self, n_in, n_out, **kwargs):
       
        self.neurons = [Neuron(n_in, **kwargs) for _ in range(n_out)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        #flatten the weights and biases of each neuron into one single list
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __repr__(self):
        return F"Layer of {len(self.neurons)} neuron/s, each with {len(self.neurons[0].parameters())} param/s"

class MLP(Module):
    """mlp consists of layers"""
    def __init__(self, n_in, n_outs):
       size =[n_in] + n_outs
       self.layers = [Layer(size[i], size[i+1], nonlin=(i!=len(n_outs)-1)) for i in range(len(n_outs))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]

    def __repr__(self):
        print(self.layers)
        return f"MLP with {len(self.layers)} layers"