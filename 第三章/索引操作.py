#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
Tensor支持与numpy.ndarray类似的索引操作，语法上也类似
如无特殊说明，索引出来的结果与原tensor共享内存，即修改一个，另一个会跟着修改
'''
import torch as t

a = t.randn(3,4)
'''tensor([[ 0.1986,  0.1809,  1.4662,  0.6693],
        [-0.8837, -0.0196, -1.0380,  0.2927],
        [-1.1032, -0.2637, -1.4972,  1.8135]])'''
print(a[0])			#第0行
'''tensor([0.1986, 0.1809, 1.4662, 0.6693])'''
print(a[:,0])		#第0列
'''tensor([ 0.1986, -0.8837, -1.1032])'''
print(a[0][2])		#第0行第2个元素，等价于a[0,2]
'''tensor(1.4662)'''
print(a[0][-1])		#第0行最后一个元素
'''tensor(0.6693)'''
print(a[:2,0:2])	#前两行，第0,1列
'''tensor([[ 0.1986,  0.1809],
        [-0.8837, -0.0196]])'''

print(a[0:1,:2])	#第0行，前两列
'''tensor([[0.1986, 0.1809]])'''
print(a[0,:2])		#注意两者的区别，形状不同
'''tensor([0.1986, 0.1809])'''

print(a>1)
'''tensor([[0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1]], dtype=torch.uint8)'''
print(a[a>1])		#等价于a.masked_select(a>1),选择结果与原tensor不共享内存空间
print(a.masked_select(a>1))
'''tensor([1.4662, 1.8135])
tensor([1.4662, 1.8135])'''
print(a[t.LongTensor([0,1])])
'''tensor([[ 0.1986,  0.1809,  1.4662,  0.6693],
        [-0.8837, -0.0196, -1.0380,  0.2927]])'''

'''
						常用的选择函数
index_select(input,dim,index)	在指定维度dim上选取，列如选择某些列、某些行
masked_select(input,mask)		例子如上，a[a>0],使用ByteTensor进行选取
non_zero(input)					非0元素的下标
gather(input,dim,index)			根据index，在dim维度上选取数据，输出size与index一样
gather是一个比较复杂的操作，对一个二维tensor，输出的每个元素如下：
	out[i][j] = input[index[i][j]][j]	#dim = 0
	out[i][j] = input[i][index[i][j]]	#dim = 1
'''

b = t.arange(0,16).view(4,4)
'''tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11],
        [12, 13, 14, 15]])'''
index = t.LongTensor([[0,1,2,3]])
print(b.gather(0,index))			#取对角线元素
'''tensor([[ 0,  5, 10, 15]])'''

index = t.LongTensor([[3,2,1,0]]).t()		#取反对角线上的元素
print(b.gather(1,index))
'''tensor([[ 3],
        [ 6],
        [ 9],
        [12]])'''

index = t.LongTensor([[3,2,1,0]])			#取反对角线的元素，与上面不同
print(b.gather(0,index))
'''tensor([[12,  9,  6,  3]])'''

index = t.LongTensor([[0,1,2,3],[3,2,1,0]]).t()
print(b.gather(1,index))
'''tensor([[ 0,  3],
        [ 5,  6],
        [10,  9],
        [15, 12]])'''

'''
与gather相对应的逆操作是scatter_，gather把数据从input中按index取出，而
scatter_是把取出的数据再放回去，scatter_函数时inplace操作
out = input.gather(dim,index)
out = Tensor()
out.scatter_(dim,index)
'''

x = t.rand(2, 5)
print(x)
c = t.zeros(3, 5).scatter_(0, t.LongTensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]), x)
print(c)

