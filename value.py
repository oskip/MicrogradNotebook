import math

import numpy as np


class Value:
    def __init__(self, val: float, parents=(), operator='', label='undef'):
        self.val = val
        self.operator = operator
        self.parents = parents
        self.label = label
        self.grad = 0.0
        self._backward_strat = lambda: None

    @staticmethod
    def _val_from(val):
        return val if isinstance(val, Value) else Value(val)

    def __repr__(self):
        return "Value(val=" + repr(self.val) + ")"
 
    # other + self, called as a fallback when other is e.g. a primitive and Python doesn't know what to do.
    def __radd__(self, other): 
        return self + other

    def __add__(self, other):
        val_other = self._val_from(other)
        s = "type of val_other.val " + str(type(val_other.val)) + " type of self.val " + str(type(self.val))
        child = Value(self.val + val_other.val, (self, val_other), '+')

        # This surprised me - closures in Python work differently than in Java. In Java child.grad would be 
        # "frozen" into a value when __add__() was executed. It would need to be "effectively final". This is not 
        # the case for Python, that support "real" closures.
        def add_backward_strat():
            # The derivative of other + self = 1.
            self.grad += 1.0 * child.grad
            val_other.grad += 1.0 * child.grad
        child._backward_strat = add_backward_strat
        return child

    def __sub__(self, other):
        val_other = self._val_from(other)
        return Value(self.val - val_other.val, (self, val_other), '-')
    
    def __rsub__(self, other):
        return self + -1 * other
    
    def __neg__(self):
        return -1 * self

    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * other**-1

    def __mul__(self, other):
        val_other = self._val_from(other)
        child = Value(self.val * val_other.val, (self, val_other), '*')
        def mul_backward_strat():
            # The derivative of other*self = other.
            self.grad += val_other.val * child.grad
            val_other.grad += self.val * child.grad
        child._backward_strat = mul_backward_strat
        return child

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        child = Value(self.val ** other, (self, ), f'**{other}')
        def pow_backward_strat():
            # The derivative of x ^ y over dx = y * x^(y-1) 
            self.grad += (other * self.val**(other-1)) * child.grad
        child._backward_strat = pow_backward_strat
        return child
    
    def tanh(self, label='tanh'):
        formula = np.tanh(self.val)
        child = Value(formula, (self, ), 'tanh', label)
        def tanh_backward_strat():
            self.grad = (1 - child.val**2) * child.grad
        child._backward_strat = tanh_backward_strat
        return child
    
    def backward(self):
        # I used to have a recursive implementation here, but I had trouble setting the initial gradient to 1.0
        # for the root node. This is like DFS, but with "post-order" instead of "in-order"
        self.grad = 1.0
        visited = set()
        topo = []
        def topo_sort(node):
            if node in visited:
                return
            visited.add(node)
            for parent in node.parents:
                topo_sort(parent)
            topo.append(node)
        topo_sort(self)
        for node in topo[::-1]:
            node._backward_strat()