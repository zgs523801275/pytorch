>>> import torch
>>> import torchvision.models as models
>>> googlenet = models.goolenet()
>>> googlenet = models.googlenet()
>>> googlenet
#GoogleNet模型采用inception结构来组成网络，总共使用了9个inception(3a,3b,4a,4b,4c,4d,4e,5a,5b)
#其中4a和4d后分别增加了aux结构，用于增强向前传递梯度
GoogLeNet(
  #层1：输入通道：3, 输出通道：64, 卷积核：7*7, 卷积个数：3*64
  (conv1): BasicConv2d(
    (conv): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
  )
  (maxpool1): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
  #层2：输入通道：64, 输出通道：64, 卷积核：1*1, 卷积个数：64*64
  (conv2): BasicConv2d(
    (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
  )
  #层3：输入通道：64, 输出通道：192, 卷积核：3*3, 卷积个数：64*192
  (conv3): BasicConv2d(
    (conv): Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
  )
  (maxpool2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
  #inception3a(层4~层5)：其中各个branch分别接受上层的特征输出，并行处理，最后通过torch.cat(按行维度)合并重组
  #branch1包含1个层：输入通道：192, 输出通道：64, 卷积核：1*1, 卷积个数：192*64
  #branch2包含2个层：输入通道：192, 输出通道：96, 卷积核：1*1, 卷积个数：192*96
  #                  输入通道：96, 输出通道：128, 卷积核：3*3, 卷积个数：96*128
  #branch3包含2个层：输入通道：192, 输出通道：16, 卷积核：1*1, 卷积个数：192*64
  #                  输入通道：16, 输出通道：32, 卷积核：3*3, 卷积个数：192*64
  #branch4包含1个层：输入通道：192, 输出通道：32, 卷积核：1*1, 卷积个数：192*64
  (inception3a): Inception(
    (branch1): BasicConv2d(
      (conv): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch2): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(192, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (branch3): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(192, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (branch4): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
      (1): BasicConv2d(
        (conv): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
  #inception3b(层6~层7)：其中各个branch分别接受上层的特征输出，并行处理，最后通过torch.cat(按行维度)合并重组
  #branch1包含1个层：输入通道：256, 输出通道：128, 卷积核：1*1, 卷积个数：256*128
  #branch2包含2个层：输入通道：256, 输出通道：128, 卷积核：1*1, 卷积个数：192*96
  #                  输入通道：128, 输出通道：192, 卷积核：3*3, 卷积个数：128*192
  #branch3包含2个层：输入通道：256, 输出通道：32, 卷积核：1*1, 卷积个数：256*32
  #                  输入通道：32, 输出通道：96, 卷积核：3*3, 卷积个数：32*96
  #branch4包含1个层：输入通道：256, 输出通道：64, 卷积核：1*1, 卷积个数：256*64
  (inception3b): Inception(
    (branch1): BasicConv2d(
      (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch2): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(128, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (branch3): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(32, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (branch4): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
      (1): BasicConv2d(
        (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
  (maxpool3): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
  #inception4a(层8~层9)：其中各个branch分别接受上层的特征输出，并行处理，最后通过torch.cat(按行维度)合并重组
  #branch1包含1个层：输入通道：480, 输出通道：192, 卷积核：1*1, 卷积个数：480*192
  #branch2包含2个层：输入通道：480, 输出通道：96, 卷积核：1*1, 卷积个数：480*96
  #                  输入通道：96, 输出通道：208, 卷积核：3*3, 卷积个数：96*208
  #branch3包含2个层：输入通道：480, 输出通道：16, 卷积核：1*1, 卷积个数：480*16
  #                  输入通道：16, 输出通道：48, 卷积核：3*3, 卷积个数：16*48
  #branch4包含1个层：输入通道：480, 输出通道：64, 卷积核：1*1, 卷积个数：480*64
  (inception4a): Inception(
    (branch1): BasicConv2d(
      (conv): Conv2d(480, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch2): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(480, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(96, 208, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(208, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (branch3): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(480, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(16, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (branch4): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
      (1): BasicConv2d(
        (conv): Conv2d(480, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
  #inception4b(层10~层11)：其中各个branch分别接受上层的特征输出，并行处理，最后通过torch.cat(按行维度)合并重组
  #branch1包含1个层：输入通道：512, 输出通道：160, 卷积核：1*1, 卷积个数：512*160
  #branch2包含2个层：输入通道：512, 输出通道：112, 卷积核：1*1, 卷积个数：512*112
  #                  输入通道：112, 输出通道：224, 卷积核：3*3, 卷积个数：112*224
  #branch3包含2个层：输入通道：512, 输出通道：24, 卷积核：1*1, 卷积个数：512*24
  #                  输入通道：24, 输出通道：64, 卷积核：3*3, 卷积个数：24*64
  #branch4包含1个层：输入通道：512, 输出通道：64, 卷积核：1*1, 卷积个数：512*64
  (inception4b): Inception(
    (branch1): BasicConv2d(
      (conv): Conv2d(512, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch2): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(512, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(112, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(112, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(224, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (branch3): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(512, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(24, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (branch4): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
      (1): BasicConv2d(
        (conv): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
  #inception4c(层12~层13)：其中各个branch分别接受上层的特征输出，并行处理，最后通过torch.cat(按行维度)合并重组
  #branch1包含1个层：输入通道：512, 输出通道：128, 卷积核：1*1, 卷积个数：512*128
  #branch2包含2个层：输入通道：512, 输出通道：128, 卷积核：1*1, 卷积个数：512*128
  #                  输入通道：128, 输出通道：256, 卷积核：3*3, 卷积个数：128*256
  #branch3包含2个层：输入通道：512, 输出通道：24, 卷积核：1*1, 卷积个数：512*24
  #                  输入通道：24, 输出通道：64, 卷积核：3*3, 卷积个数：24*64
  #branch4包含1个层：输入通道：512, 输出通道：64, 卷积核：1*1, 卷积个数：512*64
  (inception4c): Inception(
    (branch1): BasicConv2d(
      (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch2): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (branch3): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(512, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(24, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (branch4): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
      (1): BasicConv2d(
        (conv): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
  #inception4d(层14~层15)：其中各个branch分别接受上层的特征输出，并行处理，最后通过torch.cat(按行维度)合并重组
  #branch1包含1个层：输入通道：512, 输出通道：112, 卷积核：1*1, 卷积个数：512*112
  #branch2包含2个层：输入通道：512, 输出通道：144, 卷积核：1*1, 卷积个数：512*144
  #                  输入通道：144, 输出通道：288, 卷积核：3*3, 卷积个数：144*288
  #branch3包含2个层：输入通道：512, 输出通道：32, 卷积核：1*1, 卷积个数：512*32
  #                  输入通道：32, 输出通道：64, 卷积核：3*3, 卷积个数：32*64
  #branch4包含1个层：输入通道：512, 输出通道：64, 卷积核：1*1, 卷积个数：512*64
  (inception4d): Inception(
    (branch1): BasicConv2d(
      (conv): Conv2d(512, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(112, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch2): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(512, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(144, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(144, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(288, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (branch3): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(512, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (branch4): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
      (1): BasicConv2d(
        (conv): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
  #inception4e(层16~层17)：其中各个branch分别接受上层的特征输出，并行处理，最后通过torch.cat(按行维度)合并重组
  #branch1包含1个层：输入通道：528, 输出通道：256, 卷积核：1*1, 卷积个数：528*256
  #branch2包含2个层：输入通道：528, 输出通道：160, 卷积核：1*1, 卷积个数：528*160
  #                  输入通道：160, 输出通道：320, 卷积核：3*3, 卷积个数：160*320
  #branch3包含2个层：输入通道：528, 输出通道：32, 卷积核：1*1, 卷积个数：528*32
  #                  输入通道：32, 输出通道：128, 卷积核：3*3, 卷积个数：32*128
  #branch4包含1个层：输入通道：528, 输出通道：128, 卷积核：1*1, 卷积个数：528*128
  (inception4e): Inception(
    (branch1): BasicConv2d(
      (conv): Conv2d(528, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch2): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(528, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(160, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (branch3): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(528, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (branch4): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
      (1): BasicConv2d(
        (conv): Conv2d(528, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
  (maxpool4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
  #inception5a(层18~层19)：其中各个branch分别接受上层的特征输出，并行处理，最后通过torch.cat(按行维度)合并重组
  #branch1包含1个层：输入通道：832, 输出通道：256, 卷积核：1*1, 卷积个数：832*256
  #branch2包含2个层：输入通道：832, 输出通道：160, 卷积核：1*1, 卷积个数：832*160
  #                  输入通道：160, 输出通道：320, 卷积核：3*3, 卷积个数：160*320
  #branch3包含2个层：输入通道：832, 输出通道：32, 卷积核：1*1, 卷积个数：832*32
  #                  输入通道：32, 输出通道：128, 卷积核：3*3, 卷积个数：32*128
  #branch4包含1个层：输入通道：832, 输出通道：128, 卷积核：1*1, 卷积个数：832*128
  (inception5a): Inception(
    (branch1): BasicConv2d(
      (conv): Conv2d(832, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch2): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(832, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(160, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (branch3): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(832, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (branch4): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
      (1): BasicConv2d(
        (conv): Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
  #inception5b(层20~层21)：其中各个branch分别接受上层的特征输出，并行处理，最后通过torch.cat(按行维度)合并重组
  #branch1包含1个层：输入通道：832, 输出通道：384, 卷积核：1*1, 卷积个数：832*384
  #branch2包含2个层：输入通道：832, 输出通道：192, 卷积核：1*1, 卷积个数：832*192
  #                  输入通道：192, 输出通道：384, 卷积核：3*3, 卷积个数：192*384
  #branch3包含2个层：输入通道：832, 输出通道：48, 卷积核：1*1, 卷积个数：832*48
  #                  输入通道：48, 输出通道：128, 卷积核：3*3, 卷积个数：48*128
  #branch4包含1个层：输入通道：832, 输出通道：128, 卷积核：1*1, 卷积个数：832*128
  (inception5b): Inception(
    (branch1): BasicConv2d(
      (conv): Conv2d(832, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch2): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(832, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (branch3): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(832, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(48, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (branch4): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
      (1): BasicConv2d(
        (conv): Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
  #aux1：位于inception4a之后，用于增强反向梯度传递
  (aux1): InceptionAux(
    (conv): BasicConv2d(
      (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (fc1): Linear(in_features=2048, out_features=1024, bias=True)
    (fc2): Linear(in_features=1024, out_features=1000, bias=True)
  )
  #aux2：位于inception4d之后，用于增强反向梯度传递
  (aux2): InceptionAux(
    (conv): BasicConv2d(
      (conv): Conv2d(528, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (fc1): Linear(in_features=2048, out_features=1024, bias=True)
    (fc2): Linear(in_features=1024, out_features=1000, bias=True)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (dropout): Dropout(p=0.2)
  #层22(全连接层)：输入通道：1024, 输出通道：1000
  (fc): Linear(in_features=1024, out_features=1000, bias=True)
)