import math

class Value:
  
  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.grad = 0.0     #gradient at each node
    self._backward = lambda: None   #function to find chain rule at each node, chaining output gradient to input gradient
    self._prev = set(_children)
    self._op = _op
    self.label = label

  def __repr__(self):
    return f"Value(data={self.data})"
  
  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other) #wrap any additional numbers as a Value if it is not of value type already
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
        self.grad += 1.0 * out.grad #for addition in backprop, the gradient just passed down from the previous output for both nodes in the addition
        other.grad += 1.0 * out.grad
    out._backward = _backward
      
    return out

  def __mul__(self, other):
    
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad
    out._backward = _backward
    return out

  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    out = Value(self.data**other, (self,), f'**{other}') #raise self value to int or float

    def _backward():
        self.grad +=  other * (self.data ** (other - 1)) * out.grad
    out._backward = _backward

    return out

  def __rmul__(self, other):     #checking if we can multiply the other way for specific mult cases
      return self * other
      
  def __radd__(self, other):
      return self + other
      
  def __truediv__(self, other):
      return self * other**-1  #implementing division as multiplication with reciprocal

  def __neg__(self): # negate self value
    return self * -1

  def __sub__(self, other): # self - other
    return self + (-other)

  def tanh(self):
      x = self.data
      t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
      out = Value(t, (self, ), 'tanh')

      def _backward():
          self.grad += (1 - t**2) * out.grad  #think output gradient passed down (multiplied by local function gradient), to get the final self gradient value
         
      out._backward = _backward
      return out

  def exp(self):
      x = self.data
      out = Value(math.exp(x), (self, ), 'exp')
    
      def _backward():
        self.grad += out.data * out.grad #out.data is e^x, which is the derivative of e^x
      out._backward = _backward
    
      return out
      
  def backward(self):

    topo = []    #implementing topological sort going from leaf -> o, chilren first, then parent node
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)   #topoligical order in topo list of each node
    topo

    self.grad = 1.0     #top value gradient is always 1
    for node in reversed(topo):     #in backprop order, we setup gradients 
        node._backward()
