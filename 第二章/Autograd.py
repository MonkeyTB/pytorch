#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:utf-8 -*-
import torch as t
from  torch.autograd import Variable

x = Variable(t.ones(2,2),requires_grad = True)
print(x)
'''tensor([[1., 1.],
        [1., 1.]], requires_grad=True)'''
y = x.sum()
print(y)
'''tensor(4., grad_fn=<SumBackward0>)'''
print(y.grad_fn)	#指向一个Function对象，这个Function用来反向传播计算输入的梯度
'''<SumBackward0 object at 0x000002D4240AB860>'''
y.backward()
print(x.grad)
'''tensor([[1., 1.],
        [1., 1.]])'''
y.backward()
print(x.grad)
'''tensor([[2., 2.],
        [2., 2.]])'''
y.backward()
print( x.grad )
'''tensor([[3., 3.],
        [3., 3.]])'''
'''grad在反向传播过程中时累加的(accumulated)，这意味着运行
反向传播，梯度都会累加之前的梯度，所以反向传播之前需要梯度清零'''
print( x.grad.data.zero_() )
'''tensor([[0., 0.],
        [0., 0.]])'''

y.backward()
print( x.grad )
'''tensor([[1., 1.],
        [1., 1.]])'''

m = Variable(t.ones(4,5))
n = t.cos(m)
print(m)
print(n)
'''tensor([[1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.]])
tensor([[0.5403, 0.5403, 0.5403, 0.5403, 0.5403],
        [0.5403, 0.5403, 0.5403, 0.5403, 0.5403],
        [0.5403, 0.5403, 0.5403, 0.5403, 0.5403],
        [0.5403, 0.5403, 0.5403, 0.5403, 0.5403]])'''
m_tensor_cos = t.cos(m.data)
print(m_tensor_cos)
'''ensor([[0.5403, 0.5403, 0.5403, 0.5403, 0.5403],
        [0.5403, 0.5403, 0.5403, 0.5403, 0.5403],
        [0.5403, 0.5403, 0.5403, 0.5403, 0.5403],
        [0.5403, 0.5403, 0.5403, 0.5403, 0.5403]])'''
