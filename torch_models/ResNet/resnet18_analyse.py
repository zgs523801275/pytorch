>>> import torch
>>> import torchvision.models as models
>>> resnet18 = models.resnet18()
>>> resnet18
ResNet(
  #层1：输入通道：3, 输出通道：64, 卷积核：7*7, 卷积个数：3*64
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  #resnet18模型使用BasicBlock块(由2个3*3的卷积组成)来组成深层网络，采用残差学习的方式，提升精度
  #具体：下一个BasicBlock块的输入 = 上一个BasicBlock块的无处理或经下采样处理的输入 + 经上一个BasicBlock块处理的输出
  #为了残差学习，若通道匹配(如layer1)，则不需要进行下采样处理，否则需要进行下采样处理(在每个BasicBlock(0)进行下采样)，实现通道匹配
  #采用的卷积参数(kernal_size/stride/padding)保证了输入输出大小不变

  #序列层1(层2~层4):使用BasicBlock块增加残差学习项来组成
  (layer1): Sequential(
    #BasicBlock(0)对应输入：层1_输出
    (0): BasicBlock(
      #层2：输入通道：64, 输出通道：64, 卷积核：3*3, 卷积个数：64*64
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      #层3：输入通道：64, 输出通道：64, 卷积核：3*3, 卷积个数：64*64
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    #BasicBlock(1)对应输入：层2_输入 + 层3_输出
    (1): BasicBlock(
      #层4：输入通道：64, 输出通道：64, 卷积核：3*3, 卷积个数：64*64
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      #层5：输入通道：64, 输出通道：64, 卷积核：3*3, 卷积个数：64*64
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  #序列层2(层6~层9):使用BasicBlock块(增加残差学习项)来组成
  (layer2): Sequential(
    #BasicBlock(0)对应输入：层4_输入 + 层5_输出
    (0): BasicBlock(
      #层6：输入通道：64, 输出通道：128, 卷积核：3*3, 卷积个数：64*128
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      #层7：输入通道：128, 输出通道：128, 卷积核：3*3, 卷积个数：128*128
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      #下采样层：由于layer1的输出通道为64，为了实现残差学习，需要采用1*1卷积将其通道转换为128
      (downsample): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    #BasicBlock(1)对应输入：层6_输入(经下采样处理) + 层7_输出
    (1): BasicBlock(
      #层8：输入通道：128, 输出通道：128, 卷积核：3*3, 卷积个数：128*128
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      #层9：输入通道：128, 输出通道：128, 卷积核：3*3, 卷积个数：128*128
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  #序列层3(层10~层13):使用BasicBlock块(增加残差学习项)来组成
  (layer3): Sequential(
    #BasicBlock(0)对应输入：层8_输入 + 层9_输出
    (0): BasicBlock(
      #层10：输入通道：128, 输出通道：256, 卷积核：3*3, 卷积个数：128*256
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      #层11：输入通道：256, 输出通道：256, 卷积核：3*3, 卷积个数：256*256
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      #下采样层：由于layer2的输出通道为128，为了实现残差学习，需要采用1*1卷积将其通道转换为256
      (downsample): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    #BasicBlock(1)对应输入：层10_输入(经下采样处理) + 层11_输出
    (1): BasicBlock(
      #层12：输入通道：256, 输出通道：256, 卷积核：3*3, 卷积个数：256*256
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      #层13：输入通道：256, 输出通道：256, 卷积核：3*3, 卷积个数：256*256
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  #序列层4(层14~层17):使用BasicBlock块(增加残差学习项)来组成
  (layer4): Sequential(
    #BasicBlock(0)对应输入：层12_输入 + 层13_输出
    (0): BasicBlock(
      #层14：输入通道：256, 输出通道：512, 卷积核：3*3, 卷积个数：256*512
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      #层15：输入通道：512, 输出通道：512, 卷积核：3*3, 卷积个数：512*512
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      #下采样层：由于layer3的输出通道为256，为了实现残差学习，需要采用1*1卷积将其通道转换为512
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    #BasicBlock(1)对应输入：层14_输入(经下采样层处理) + 层15_输出
    (1): BasicBlock(
      #层16：输入通道：512, 输出通道：512, 卷积核：3*3, 卷积个数：512*512
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      #层17：输入通道：512, 输出通道：512, 卷积核：3*3, 卷积个数：512*512
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  #全连接层对应输入：层16_输入 + 层17_输出
  #层18：输入通道：512, 输出通道：1000
  (fc): Linear(in_features=512, out_features=1000, bias=True)
)