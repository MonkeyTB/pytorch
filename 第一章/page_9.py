#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:utf-8 -*-
import torch
from torch.autograd import Variable

N,D,H = 3,4,5

x = Variable(torch.randn(N,D))		#生成随机数并转为Tensor格式
w1 = Variable(torch.randn(D,H))
w2 = Variable(torch.randn(D,H))

z = 10
if z > 0:
	y = x.mm(w1)		#矩阵点乘
else:
	y = x.mm(w2)
print(y)