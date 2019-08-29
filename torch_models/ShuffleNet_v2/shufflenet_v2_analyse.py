>>> import torch
>>> import torchvision.models as models
>>> shufflenet_v2_x0_5 = models.shufflenet_v2_x0_5()
>>> shufflenet_v2_x0_5
#shufflenet_v2_x0_5主要由InvertedResidual结构(模型特色)组成
#其中每个stage包含若干个InvertedResidual结构：
#stage2包含4个InvertedResidual结构：InvertedResidual_0/InvertedResidual_1/InvertedResidual_2/InvertedResidual_3
#InvertedResidual_0包含2个branch：branch1(3*3卷积 + 1*1卷积), branch2(1*1卷积 + 3*3卷积 + 1*1卷积)
#InvertedResidual_1/InvertedResidual_2/InvertedResidual_3均包含2个branch：branch1(直连), branch2(1*1卷积 + 3*3卷积 + 1*1卷积)
#对于branch1和branch2的输出通过cat合并，在使用channel shuffle进行特征重组(模型特色)，输入到下一级InvertedResidual中
#stage3包含7个InvertedResidual结构：相较stage2增加了像InvertedResidual_1这样的结构
#stage4包含4个InvertedResidual结构：与stage2相似

#对于shufflenet_v2_x1_0/shufflenet_v2_x1_5/shufflenet_v2_x2_0：
#相较于shufflenet_v2_x0_5, 其结构未发生任何变化，只是各InvertedResidual结构输入输出通道有所变化
ShuffleNetV2(
  (conv1): Sequential(
    (0): Conv2d(3, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace)
  )
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (stage2): Sequential(
    (0): InvertedResidual(
      (branch1): Sequential(
        (0): Conv2d(24, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=24, bias=False)
        (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (3): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (4): ReLU(inplace)
      )
      (branch2): Sequential(
        (0): Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace)
        (3): Conv2d(24, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=24, bias=False)
        (4): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (6): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (7): ReLU(inplace)
      )
    )
    (1): InvertedResidual(
      (branch2): Sequential(
        (0): Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace)
        (3): Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24, bias=False)
        (4): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (6): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (7): ReLU(inplace)
      )
    )
    (2): InvertedResidual(
      (branch2): Sequential(
        (0): Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace)
        (3): Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24, bias=False)
        (4): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (6): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (7): ReLU(inplace)
      )
    )
    (3): InvertedResidual(
      (branch2): Sequential(
        (0): Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace)
        (3): Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24, bias=False)
        (4): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (6): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (7): ReLU(inplace)
      )
    )
  )
  (stage3): Sequential(
    (0): InvertedResidual(
      (branch1): Sequential(
        (0): Conv2d(48, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=48, bias=False)
        (1): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (3): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (4): ReLU(inplace)
      )
      (branch2): Sequential(
        (0): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace)
        (3): Conv2d(48, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=48, bias=False)
        (4): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (6): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (7): ReLU(inplace)
      )
    )
    (1): InvertedResidual(
      (branch2): Sequential(
        (0): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace)
        (3): Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=48, bias=False)
        (4): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (6): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (7): ReLU(inplace)
      )
    )
    (2): InvertedResidual(
      (branch2): Sequential(
        (0): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace)
        (3): Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=48, bias=False)
        (4): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (6): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (7): ReLU(inplace)
      )
    )
    (3): InvertedResidual(
      (branch2): Sequential(
        (0): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace)
        (3): Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=48, bias=False)
        (4): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (6): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (7): ReLU(inplace)
      )
    )
    (4): InvertedResidual(
      (branch2): Sequential(
        (0): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace)
        (3): Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=48, bias=False)
        (4): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (6): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (7): ReLU(inplace)
      )
    )
    (5): InvertedResidual(
      (branch2): Sequential(
        (0): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace)
        (3): Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=48, bias=False)
        (4): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (6): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (7): ReLU(inplace)
      )
    )
    (6): InvertedResidual(
      (branch2): Sequential(
        (0): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace)
        (3): Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=48, bias=False)
        (4): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (6): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (7): ReLU(inplace)
      )
    )
    (7): InvertedResidual(
      (branch2): Sequential(
        (0): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace)
        (3): Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=48, bias=False)
        (4): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (6): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (7): ReLU(inplace)
      )
    )
  )
  (stage4): Sequential(
    (0): InvertedResidual(
      (branch1): Sequential(
        (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)
        (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (3): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (4): ReLU(inplace)
      )
      (branch2): Sequential(
        (0): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace)
        (3): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)
        (4): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (6): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (7): ReLU(inplace)
      )
    )
    (1): InvertedResidual(
      (branch2): Sequential(
        (0): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace)
        (3): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
        (4): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (6): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (7): ReLU(inplace)
      )
    )
    (2): InvertedResidual(
      (branch2): Sequential(
        (0): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace)
        (3): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
        (4): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (6): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (7): ReLU(inplace)
      )
    )
    (3): InvertedResidual(
      (branch2): Sequential(
        (0): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace)
        (3): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
        (4): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (6): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (7): ReLU(inplace)
      )
    )
  )
  (conv5): Sequential(
    (0): Conv2d(192, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace)
  )
  (fc): Linear(in_features=1024, out_features=1000, bias=True)
)