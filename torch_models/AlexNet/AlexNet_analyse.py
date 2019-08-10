>>> import torch
>>> import torchvision.models as models
>>> alexnet = models.alexnet()
>>> alexnet
AlexNet(
  (features): Sequential(
    #层1：输入通道：3, 输出通道：64, 卷积核：11*11, 卷积个数：3*64
    (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    (1): ReLU(inplace)#激活函数
    #重叠池化层：3 > 2 
    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    #层2：输入通道：64, 输出通道：192, 卷积核：5*5, 卷积个数：64*192
    (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): ReLU(inplace)#激活函数(inplace=True:直接修改上层传递的数据，节省内存空间)
    #重叠池化层：3 > 2 
    (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    #层3：输入通道：192, 输出通道：384, 卷积核：3*3, 卷积个数：192*384
    (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace)#激活函数
    #层4：输入通道：384, 输出通道：256, 卷积核：3*3, 卷积个数：384*256
    (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(inplace)#激活函数
    #层5：输入通道：256, 输出通道：256, 卷积核：3*3, 卷积个数：256*256
    (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace)#激活函数
    #重叠池化层：3 > 2 
    (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  #2维自适应均值池化层，确保输出特征大小为6*6
  #stride = floor(input_size / (output_size - 1))
  #kernel_size = input_size - (output_size - 1) * stride
  (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
  (classifier): Sequential(
    #Dropout层：随机忽略一半的神经元(p=0.5)
    (0): Dropout(p=0.5)
    #层6：输入通道：9216(256*6*6), 输出通道：9216
    (1): Linear(in_features=9216, out_features=4096, bias=True)
    (2): ReLU(inplace)#激活函数
    #Dropout层：随机忽略一半的神经元(p=0.5)
    (3): Dropout(p=0.5)
    #层7：输入通道：4096, 输出通道：4096
    (4): Linear(in_features=4096, out_features=4096, bias=True)
    (5): ReLU(inplace)#激活函数
    #层8：输入通道：4096, 输出通道：1000
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)