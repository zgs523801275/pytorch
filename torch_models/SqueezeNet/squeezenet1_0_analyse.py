>>> import torch
>>> import torchvision.models as models
>>> squeezenet1_0 = models.squeezenet1_0()
>>> squeezenet1_0
#SqueezeNet采用Fire模块取代conv+pool方式，减小了模型，并提升了精度
#Fire模块多采用1*1卷积，并附加一部分3*3卷积，能够在保证特征提取不失的同时，降低计算量
#采用全卷积的方式组建模型，抛弃线性全连接层
SqueezeNet(
  (features): Sequential(
    #层1：输入通道：3, 输出通道：96, 卷积核：7*7, 卷积个数：3*96
    (0): Conv2d(3, 96, kernel_size=(7, 7), stride=(2, 2))
    (1): ReLU(inplace)
    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
    #Fire采用squeeze(压缩：1*1卷积)和expand(膨胀：1*1卷积 + 3*3卷积)两部分组成
    #其中squeeze部分的输出分别传递给expand部分中expand1x1和expand3x3作为输入
    #expand部分中expand1x1和expand3x3的输出经torch.cat重组(按行维度)后作为下一级Fire的输入
    (3): Fire(
      #层2：输入通道：96, 输出通道：16, 卷积核：1*1, 卷积个数：96*16
      (squeeze): Conv2d(96, 16, kernel_size=(1, 1), stride=(1, 1))
      (squeeze_activation): ReLU(inplace)
      #层3：1*1卷积+3*3卷积
      #1*1卷积：输入通道：16, 输出通道：64, 卷积核：1*1, 卷积个数：16*64
      (expand1x1): Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1))
      (expand1x1_activation): ReLU(inplace)
      #3*3卷积：输入通道：16, 输出通道：64, 卷积核：3*3, 卷积个数：16*64
      (expand3x3): Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (expand3x3_activation): ReLU(inplace)
    )
    (4): Fire(
      #层4：输入通道：128, 输出通道：16, 卷积核：1*1, 卷积个数：128*16
      (squeeze): Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1))
      (squeeze_activation): ReLU(inplace)
      #层5：1*1卷积+3*3卷积
      #1*1卷积：输入通道：16, 输出通道：64, 卷积核：1*1, 卷积个数：16*64
      (expand1x1): Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1))
      (expand1x1_activation): ReLU(inplace)
      #3*3卷积：输入通道：16, 输出通道：64, 卷积核：3*3, 卷积个数：16*64
      (expand3x3): Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (expand3x3_activation): ReLU(inplace)
    )
    (5): Fire(
      #层6：输入通道：128, 输出通道：32, 卷积核：1*1, 卷积个数：128*32
      (squeeze): Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1))
      (squeeze_activation): ReLU(inplace)
      #层7：1*1卷积+3*3卷积
      #1*1卷积：输入通道：32, 输出通道：128, 卷积核：1*1, 卷积个数：32*128
      (expand1x1): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
      (expand1x1_activation): ReLU(inplace)
      #3*3卷积：输入通道：32, 输出通道：128, 卷积核：3*3, 卷积个数：32*128
      (expand3x3): Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (expand3x3_activation): ReLU(inplace)
    )
    (6): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
    (7): Fire(
      #层8：输入通道：256, 输出通道：32, 卷积核：1*1, 卷积个数：256*32
      (squeeze): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
      (squeeze_activation): ReLU(inplace)
      #层9：1*1卷积+3*3卷积
      #1*1卷积：输入通道：32, 输出通道：128, 卷积核：1*1, 卷积个数：32*128
      (expand1x1): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
      (expand1x1_activation): ReLU(inplace)
      #3*3卷积：输入通道：32, 输出通道：128, 卷积核：3*3, 卷积个数：32*128
      (expand3x3): Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (expand3x3_activation): ReLU(inplace)
    )
    (8): Fire(
      #层10：输入通道：256, 输出通道：48, 卷积核：1*1, 卷积个数：256*48
      (squeeze): Conv2d(256, 48, kernel_size=(1, 1), stride=(1, 1))
      (squeeze_activation): ReLU(inplace)
      #层11：1*1卷积+3*3卷积
      #1*1卷积：输入通道：48, 输出通道：192, 卷积核：1*1, 卷积个数：48*192
      (expand1x1): Conv2d(48, 192, kernel_size=(1, 1), stride=(1, 1))
      (expand1x1_activation): ReLU(inplace)
      #3*3卷积：输入通道：48, 输出通道：192, 卷积核：3*3, 卷积个数：48*192
      (expand3x3): Conv2d(48, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (expand3x3_activation): ReLU(inplace)
    )
    (9): Fire(
      #层12：输入通道：3,84 输出通道：48, 卷积核：1*1, 卷积个数：384*48
      (squeeze): Conv2d(384, 48, kernel_size=(1, 1), stride=(1, 1))
      (squeeze_activation): ReLU(inplace)
      #层13：1*1卷积+3*3卷积
      #1*1卷积：输入通道：48, 输出通道：192, 卷积核：1*1, 卷积个数：48*192
      (expand1x1): Conv2d(48, 192, kernel_size=(1, 1), stride=(1, 1))
      (expand1x1_activation): ReLU(inplace)
      #3*3卷积：输入通道：48, 输出通道：192, 卷积核：3*3, 卷积个数：48*192
      (expand3x3): Conv2d(48, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (expand3x3_activation): ReLU(inplace)
    )
    (10): Fire(
      #层14：输入通道：384, 输出通道：64, 卷积核：1*1, 卷积个数：384*64
      (squeeze): Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1))
      (squeeze_activation): ReLU(inplace)
      #层15：1*1卷积+3*3卷积
      #1*1卷积：输入通道：64, 输出通道：256, 卷积核：1*1, 卷积个数：64*256
      (expand1x1): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
      (expand1x1_activation): ReLU(inplace)
      #3*3卷积：输入通道：64, 输出通道：256, 卷积核：3*3, 卷积个数：64*256
      (expand3x3): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (expand3x3_activation): ReLU(inplace)
    )
    (11): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
    (12): Fire(
      #层16：输入通道：512, 输出通道：64, 卷积核：1*1, 卷积个数：512*64
      (squeeze): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
      (squeeze_activation): ReLU(inplace)
      #层17：1*1卷积+3*3卷积
      #1*1卷积：输入通道：64, 输出通道：256, 卷积核：1*1, 卷积个数：64*256
      (expand1x1): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
      (expand1x1_activation): ReLU(inplace)
      #3*3卷积：输入通道：64, 输出通道：256, 卷积核：3*3, 卷积个数：64*256
      (expand3x3): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (expand3x3_activation): ReLU(inplace)
    )
  )
  (classifier): Sequential(
    (0): Dropout(p=0.5)
    #层18：输入通道：512, 输出通道：1000, 卷积核：1*1, 卷积个数：512*1000
    (1): Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1))
    (2): ReLU(inplace)
    (3): AdaptiveAvgPool2d(output_size=(1, 1))
  )
)