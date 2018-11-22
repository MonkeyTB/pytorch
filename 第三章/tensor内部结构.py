#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:utf-8 -*-
import numpy as np
import torch as t
a = t.arange(0,6)
a.storage()
b = a.view(2,3)
b.storage()
#a和b的storage的内存地址一样，即他们是用同一个storage
print( id(b.storage) == id(a.storage) )

#a改变，b也随之改变，因为他们共享storage
a[1] = 100
print(b)

c = a[2:]
c.storage()
print(c)

#3198436924144    3198436924128,首地址差16，因为两个元素2*8，每个元素占8个字节
print(c.data_ptr())
print(a.data_ptr())

c[0] = -100
print(a)

#3个tensor共享storage
print(id( a.storage() ) == id( b.storage() ) == id( c.storage()) )


#以储存元素的个数的形式返回tensor在地城内存中的偏移量
print( a.storage_offset() )
print( b.storage_offset() )
print( c.storage_offset() )
'''0  0  2'''


print('b',b)
e = b[::1,::2]
print('e',e)
'''b tensor([[   0,  100, -100],
        [   3,    4,    5]])
e tensor([[   0, -100],
        [   3,    5]])'''

#tensor步长
print(b.stride(),e.stride())
'''(3, 1) (3, 2)'''

#判断tensor是否连续
print(e.is_contiguous())
'''False'''
f = e.contiguous()
print(f.is_contiguous())
'''True'''