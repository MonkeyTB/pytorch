#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:utf-8 -*-

import torch as t
a = t.arange(0,6)
b = a.view(2,3)		#调整tensor的形状，不会修改自身数据
c = a.view(-1,3)	#-1自动计算大小
'''
a = tensor([0, 1, 2, 3, 4, 5])
b = tensor([[0, 1, 2],
        [3, 4, 5]])
c = tensor([[0, 1, 2],
        [3, 4, 5]])'''
b = b.unsqueeze(1)		#在第一维（下标从0开始）上增加“1”
'''tensor([[[0, 1, 2]],

        [[3, 4, 5]]])'''
c = c.unsqueeze(-2)
'''tensor([[[0, 1, 2]],

        [[3, 4, 5]]])'''
d = b.view(1,1,1,2,3)
'''tensor([[[[[0, 1, 2],
           [3, 4, 5]]]]])'''
d = d.squeeze(0)	#压缩第0维的“1”
'''tensor([[[0, 1, 2],
         [3, 4, 5]]])'''
d = d.squeeze()		#所有维度为“1”的压缩
'''tensor([[0, 1, 2],
        [3, 4, 5]])'''
b.resize_(1,3)		#resize_调整size，新尺寸小于原尺寸，则之前的保留
'''tensor([[0, 1, 2]])'''
b.resize_(3,3)		#resize_调整size，新尺寸大于原尺寸，会自动分配新的内存空间
'''tensor([[                  0,                   1,                   2],
        [                  3,                   4,                   5],
        [7526747973031060338, 7306812055932138085, 8389969046800051066]])'''