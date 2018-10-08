#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:utf-8 -*-
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		self.conv1 = nn.Conv2d(1,6,5)
		self.conv2 = nn.Conv2d(6,16,5)

		self.fc1 = nn.Linear(16*5*5,120)
		self.fc2 = nn.Linear(120,84)
		self.fc3 = nn.Linear(84,10)
	def forward(self,x):
		x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
		x = F.max_pool2d(F.relu(self.conv2(x)),2)
		x = x.view(x.size()[0],-1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x
net = Net()
print(net)
'''Net(
  (conv1): Conv1d(1, 6, kernel_size=(5,), stride=(1,))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)'''
'''网络可学习的参数'''
params = list(net.parameters())
print(len(params))	 #10
'''返回可学习的参数及名称'''
for name,parameters in net.named_parameters():
	print(name,':',parameters.size())
'''conv1.weight : torch.Size([6, 1, 5])
conv1.bias : torch.Size([6])
conv2.weight : torch.Size([16, 6, 5, 5])
conv2.bias : torch.Size([16])
fc1.weight : torch.Size([120, 400])
fc1.bias : torch.Size([120])
fc2.weight : torch.Size([84, 120])
fc2.bias : torch.Size([84])
fc3.weight : torch.Size([10, 84])
fc3.bias : torch.Size([10])'''
input = Variable(t.randn(1,1,32,32))
out = net(input)
out.size()
print(out.size())
'''torch.Size([1, 10])'''
net.zero_grad()
out.backward(t.Tensor.float( Variable(t.ones(1,10)) ))	#反向传播
output = net(input)
target = t.Tensor.float(Variable(t.arange(0,10)) ).reshape(1,10)
criterion = nn.MSELoss()		#计算均方误差
loss = criterion(output,target)
print(loss)
'''tensor(28.8579, grad_fn=<MseLossBackward>)'''

net.zero_grad()	#把net中所有可学习参数的梯度清零
print('反向传播之前conv1.bias的梯度',net.conv1.bias.grad)
loss.backward()
print('反向传播之后conv1.bias的梯度',net.conv1.bias.grad)
'''反向传播之前conv1.bias的梯度 tensor([0., 0., 0., 0., 0., 0.])
反向传播之后conv1.bias的梯度 tensor([-0.0783,  0.0173, -0.0358, -0.0556,  0.0210, -0.0596])'''

#优化器
import torch.optim as optim
#建立一个优化器，指定要调参的参数和学习率
optimizer = optim.SGD(net.parameters(),lr = 0.01)
#梯度清零
optimizer.zero_grad()
#计算损失
output = net(input)
loss = criterion(output,target)
#反向传播
loss.backward()
#更新参数
optimizer.step()