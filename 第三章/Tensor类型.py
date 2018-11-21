#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:utf-8 -*-

import torch as t
#设置默认tensor格式,正常默认是FloatTensor，修改为IntTensor
t.set_default_tensor_type(t.DoubleTensor)
a = t.Tensor(2,3)
print(a.dtype)		#torch.float64

b = a.float()		#等价于b = a.type(t.FloatTensor)
print(b.dtype)		#torch.float32

self = t.Tensor(3, 5)
print('self',self.dtype)
tesnor = t.IntTensor(2,3)
print('tensor',tesnor.dtype)

print( self.type_as(tesnor).dtype)		#通过typt_as转化类型
'''
self torch.float64
tensor torch.int32
torch.int32
'''

d = a.new(2,3)		# 等价于torch.DoubleTensor(2,3)
print('d',d)
