#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:utf-8 -*-
import torch as t
# from __future__ import print_function

a = t.Tensor(2,3)		#指定Tensor的形状，a的数值取决于内存空间的状态
print(a)
'''tensor([[2.1469e+33, 5.9555e-43, 2.1479e+33],
        [5.9555e-43, 6.3273e+30, 5.9555e-43]])'''
b = t.Tensor([ [1,2,3],[4,5,6] ])		#用list的数据创建Tensor
print(b)
'''tensor([[1., 2., 3.],
        [4., 5., 6.]])'''
c = b.tolist()		#Tensor转list
print(c)
'''[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]'''
#torch.size()返回torch.Size的子类，但其使用方式与tuple稍有区别
b_size = b.size()
print(b_size)
'''torch.Size([2, 3])'''
d = b.numel()
print(d)
'''6'''
#创建一个和b形状一样的tensor
e = t.Tensor(b_size)
f = t.Tensor((2,3))
print(e)
print(f)
'''tensor([[2.6492e+21, 4.5908e-41, 0.0000e+00],
        [0.0000e+00, 1.4013e-45, 2.9775e-41]])
tensor([2., 3.])'''
print(e.shape)
'''torch.Size([2, 3])'''
print(t.ones(2,3))
'''tensor([[1., 1., 1.],
        [1., 1., 1.]])'''
print(t.zeros(2,3))
'''tensor([[0., 0., 0.],
        [0., 0., 0.]])'''
print(t.arange(1,6,2))
'''tensor([1, 3, 5])'''
print(t.linspace(1,10,3))
'''tensor([ 1.0000,  5.5000, 10.0000])'''
print(t.randn(2,3))
'''tensor([[-0.3437, -0.3981, -0.3250],
        [ 2.6717, -0.7511, -0.5858]])'''
print(t.randperm(5))		#长度为5的随机排序
'''tensor([4, 0, 3, 2, 1])'''
print(t.eye(2,3))			#对角线为1，不要求行列数一致
'''tensor([[1., 0., 0.],
        [0., 1., 0.]])'''