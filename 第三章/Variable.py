#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:utf-8 -*-

from __future__ import print_function
import torch as t
from torch.autograd import Variable as V

a = V(t.ones(3,4),requires_grad = True)
# print(a)
b = V(t.zeros(3,4))
# print(b)
c = a.add(b)
# print(c)
d = c.sum()
# print(c)
print(c.data.sum(),c.sum())
e = a.grad
print(e)
print(a.requires_grad,b.requires_grad,c.requires_grad)
print(c.grad is None)


def f(x):
	y = x**2*t.exp(x)
	return y
def gradf(x):
	dx = 2*x*t.exp(x) + x**2*t.exp(x)
	return dx
x = V(t.randn(3,4),requires_grad = True)
y = f(x)
# print(y)
y.backward(t.ones(y.size()))
print(x.grad)	#autograd计算梯度
print(gradf(x))	#手动计算梯度