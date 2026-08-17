class Value:
    """single scalar wrapper with autograd functionality storing the gradient by applying chain rule to upstream and local gradient"""

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda: None 
        #leaf nodes has no backward, for other nodes as result of operations, store the rule for applying upstream gradient to the local gradients
        #we can break down complex operations to atomic operations (+,- etc.), but it is not necessarily needed, only we need is to know the derivate of the (specific) operation for local gradient
        #as long as you can forward pass and backward pass through an 

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value((self.data+other.data), (self, other), '+')

        def _backward():
            self.grad += 1. * out.grad
            other.grad += 1. * out.grad

        out._backward = _backward
        
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value((self.data*other.data), (self, other),'*')

        def _backward():
            self.grad += out.grad*other.data
            other.grad += out.grad*self.data
    
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)) 
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += out.grad * (other*(self.data**(other-1)))

        out._backward = _backward

        return out
  
    def relu(self):
        x = self.data
        out = Value(0 if x < 0 else x, (self, ), "ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def backward(self):
        self.grad = 1.

        topological = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topological.append(v)

        build_topo(self)

        for node in reversed(topological):
            node._backward()

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return other * self**-1
    
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"