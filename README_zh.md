# Albumentations

Albumentations是一个用于图像增强的Python库。图像增强在深度学习和计算机视觉任务中用于提高训练模型的质量。图像增强的目的是从现有数据中创建新的训练样本。

以下是一个示例，展示如何从Albumentations应用一些[像素级](#pixel-level-transforms)增强来创建从原始图像生成的新图像：
![parrot](https://habrastorage.org/webt/bd/ne/rv/bdnerv5ctkudmsaznhw4crsdfiw.jpeg)

## 为什么选择 Albumentations

- Albumentations **[支持所有常见的计算机视觉任务](#i-want-to-use-albumentations-for-the-specific-task-such-as-classification-or-segmentation)** 如分类、语义分割、实例分割、目标检测和姿态估计。
- 该库提供**[一个简单的统一API](#a-simple-example)**来处理所有数据类型：图像（RGB图像、灰度图像、多光谱图像）、分割掩码、边界框和关键点。
- 该库包含**[超过70种不同的增强](#list-of-augmentations)**来从现有数据生成新的训练样本。
- Albumentations是[**快速的**](#benchmarking-results)。我们会对每个新版本进行基准测试，以确保增强提供最大的速度。
- 它**[与流行的深度学习框架兼容](#i-want-to-know-how-to-use-albumentations-with-deep-learning-frameworks)**，如PyTorch和TensorFlow。顺便说一下，Albumentations是[PyTorch生态系统](https://pytorch.org/ecosystem/)的一部分。
- [**由专家编写**](#authors)。作者既有在生产计算机视觉系统上工作的经验，也参与了竞争激烈的机器学习。许多核心团队成员是Kaggle大师和特级大师。
- 该库在工业界、深度学习研究、机器学习竞赛和开源项目中[**广泛使用**](#who-is-using-albumentations)。

## 目录

- [Albumentations](#albumentations)
  - [为什么选择Albumentations](#why-albumentations)
  - [目录](#table-of-contents)
  - [安装](#installation)
  - [文档](#documentation)
  - [一个简单的示例](#a-simple-example)
  - [入门](#getting-started)
    - [图像增强的新手](#i-am-new-to-image-augmentation)
    - [将Albumentations用于特定任务，如分类或分割](#i-want-to-use-albumentations-for-the-specific-task-such-as-classification-or-segmentation)
    - [如何将Albumentations与深度学习框架一起使用](#i-want-to-know-how-to-use-albumentations-with-deep-learning-frameworks)
    - [探索增强并看到Albumentations的实际操作](#i-want-to-explore-augmentations-and-see-albumentations-in-action)
  - [谁在使用Albumentations](#who-is-using-albumentations)
    - [另见](#see-also)
  - [增强列表](#list-of-augmentations)
    - [像素级变换](#pixel-level-transforms)
    - [空间级变换](#spatial-level-transforms)
    - [混合级变换](#mixing-level-transforms)
  - [更多增强的例子](#a-few-more-examples-of-augmentations)
    - [在Inria数据集上的语义分割](#semantic-segmentation-on-the-inria-dataset)
    - [医学成像](#medical-imaging)
    - [在Mapillary Vistas数据集上的目标检测和语义分割](#object-detection-and-semantic-segmentation-on-the-mapillary-vistas-dataset)
    - [关键点增强](#keypoints-augmentation)
  - [基准测试结果](#benchmarking-results)
  - [贡献](#contributing)
  - [社区和支持](#community-and-support)
  - [评论](#comments)
  - [引用](#citing)


## 安装

Albumentations要求使用Python 3.8或更高版本。要从PyPI安装最新版本：

```bash
pip install -U albumentations
```

其他安装选项在[文档](https://albumentations.ai/docs/getting_started/installation/)中有描述。

## 文档

完整文档可在 **[https://albumentations.ai/docs/](https://albumentations.ai/docs/)** 查看。

## 一个简单的示例

```python
import albumentations as A
import cv2

# 声明一个增强流程
transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

# 使用OpenCV读取图片并转换为RGB颜色空间
image = cv2.imread("image.jpg")
image = cv2.cvtColor(image, cv2.COLOR

_BGR2RGB)

# 增强图片
transformed = transform(image=image)
transformed_image = transformed["image"]
```

## 入门

### 图像增强的新手

请从关于为什么图像增强重要以及它如何帮助构建更好的模型的[介绍文章](https://albumentations.ai/docs/#introduction-to-image-augmentation)开始。

### 将Albumentations用于特定任务，如分类或分割

如果您想将Albumentations用于诸如分类、分割或目标检测等特定任务，请参阅[一系列文章](https://albumentations.ai/docs/#getting-started-with-albumentations)，其中详细描述了这项任务。我们还有一个关于如何将Albumentations应用于不同用例的[示例列表](https://albumentations.ai/docs/examples/)。

### 如何将Albumentations与深度学习框架一起使用

我们有[使用Albumentations的示例](https://albumentations.ai/docs/#examples-of-how-to-use-albumentations-with-different-deep-learning-frameworks)与PyTorch和TensorFlow一起使用。

### 探索增强并看到Albumentations的实际操作

查看[库的在线演示](https://albumentations-demo.herokuapp.com/)。通过它，您可以对不同的图像应用增强并查看结果。此外，我们还有[所有可用增强及其目标的列表](#list-of-augmentations)。



## 谁在使用Albumentations

- [引用Albumentations的论文列表](https://scholar.google.com/citations?view_op=view_citation&citation_for_view=vkjh9X0AAAAJ:r0BpntZqJG4C).
- [在机器学习竞赛中使用Albumentations并取得高名次的团队列表](https://albumentations.ai/whos_using#competitions).
- [使用Albumentations的开源项目](https://github.com/albumentations-team/albumentations/network/dependents?dependent_type=PACKAGE).



## 增强列表

### 像素级变换

像素级变换只作用于输入图像，不会改变 Target (Label)，如 Masks、BBoxs 和 Keypoints。像素级变换列表：

- [AdvancedBlur（高级模糊）](https://albumentations.ai/docs/api_reference/augmentations/blur/transforms/#albumentations.augmentations.blur.transforms.AdvancedBlur)
- [Blur（模糊）](https://albumentations.ai/docs/api_reference/augmentations/blur/transforms/#albumentations.augmentations.blur.transforms.Blur)
- [CLAHE（对比度限制自适应直方图均衡化）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.CLAHE)
- [ChannelDropout（通道丢失）](https://albumentations.ai/docs/api_reference/augmentations/dropout/channel_dropout/#albumentations.augmentations.dropout.channel_dropout.ChannelDropout)
- [ChannelShuffle（通道随机交换）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ChannelShuffle)
- [ChromaticAberration（色差）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ChromaticAberration)
- [ColorJitter（颜色抖动）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ColorJitter)
- [Defocus（去焦模糊）](https://albumentations.ai/docs/api_reference/augmentations/blur/transforms/#albumentations.augmentations.blur.transforms.Defocus)
- [Downscale（降低分辨率）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Downscale)
- [Emboss（浮雕效果）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Emboss)
- [Equalize（均衡化）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Equalize)
- [FDA（频域适配）](https://albumentations.ai/docs/api_reference/augmentations/domain_adaptation/#albumentations.augmentations.domain_adaptation.FDA)
- [FancyPCA（花式PCA）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.FancyPCA)
- [FromFloat（从浮点数）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.FromFloat)
- [GaussNoise（高斯噪声）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.GaussNoise)
- [GaussianBlur（高斯模糊）](https://albumentations.ai/docs/api_reference/augmentations/blur/transforms/#albumentations.augmentations.blur.transforms.GaussianBlur)
- [GlassBlur（玻璃模糊）](https://albumentations.ai/docs/api_reference/augmentations/blur/transforms/#albumentations.augmentations.blur.transforms.GlassBlur)
- [HistogramMatching（直方图匹配）](https://albumentations.ai/docs/api_reference/augmentations/domain_adaptation/#albumentations.augmentations.domain_adaptation.HistogramMatching)
- [HueSaturationValue（色调饱和度明度）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.HueSaturationValue)
- [ISONoise（ISO噪声）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ISONoise)
- [ImageCompression（图像压缩）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ImageCompression)
- [InvertImg（反转图像）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.InvertImg)
- [MedianBlur（中值模糊）](https://albumentations.ai/docs/api_reference/augmentations/blur/transforms/#albumentations.augmentations.blur.transforms.MedianBlur)
- [MotionBlur（运动模糊）](https://albumentations.ai/docs/api_reference/augmentations/blur/transforms/#albumentations.augmentations.blur.transforms.MotionBlur)
- [MultiplicativeNoise（乘性噪声）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.MultiplicativeNoise)
- [Normalize（标准化）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Normalize)
- [PixelDistributionAdaptation（像素分布适配）](https://albumentations.ai/docs/api_reference/augmentations/domain_adaptation/#albumentations.augmentations.domain_adaptation.PixelDistributionAdaptation)
- [PlanckianJitter（普朗克抖动）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.PlanckianJitter)
- [Posterize（色阶减少）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Posterize)
- [RGBShift（RGB偏移）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RGBShift)
- [RandomBrightnessContrast（随机亮度对比度）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomBrightnessContrast)
- [RandomFog（随机雾化）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomFog)
- [RandomGamma（随机伽马）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomGamma)
- [RandomGravel（随机砾石）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomGravel)
- [RandomRain（随机雨滴）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomRain)
- [RandomShadow（随机阴影）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomShadow)

- [RandomSnow（随机雪花）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomSnow)
- [RandomSunFlare（随机太阳耀斑）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomSunFlare)
- [RandomToneCurve（随机色调曲线）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomToneCurve)
- [RingingOvershoot（振铃过冲）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RingingOvershoot)
- [Sharpen（锐化）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Sharpen)
- [Solarize（曝光过度）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Solarize)
- [Spatter（飞溅）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Spatter)
- [Superpixels（超像素）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Superpixels)
- [TemplateTransform（模板转换）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.TemplateTransform)
- [TextImage（文字图像）](https://albumentations.ai/docs/api_reference/augmentations/text/transforms/#albumentations.augmentations.text.transforms.TextImage)
- [ToFloat（转换为浮点数）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ToFloat)
- [ToGray（转换为灰度）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ToGray)
- [ToRGB（转换为RGB）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ToRGB)
- [ToSepia（转换为棕褐色）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ToSepia)
- [UnsharpMask（非锐化掩码）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.UnsharpMask)
- [ZoomBlur（缩放模糊）](https://albumentations.ai/docs/api_reference/augmentations/blur/transforms/#albumentations.augmentations.blur.transforms.ZoomBlur)



### 空间级变换

空间级变换将同时改变输入图像及其附加目标，如掩码、边界框和关键点。下表显示了每种变换支持的附加目标。

| 变换名称                                                                                                                                                                       | 图像 | 掩码 | 边界框 | 关键点 |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---: | :--: | :----: | :-------: |
| [Affine（仿射变换）](https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.Affine)                             | ✓     | ✓    | ✓      | ✓         |
| [BBoxSafeRandomCrop（边界框安全随机裁剪）](https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/#albumentations.augmentations.crops.transforms.BBoxSafeRandomCrop)             | ✓     | ✓    | ✓      | ✓         |
| [CenterCrop（中心裁剪）](https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/#albumentations.augmentations.crops.transforms.CenterCrop)                             | ✓     | ✓    | ✓      | ✓         |
| [CoarseDropout（粗略随机丢失）](https://albumentations.ai/docs/api_reference/augmentations/dropout/coarse_dropout/#albumentations.augmentations.dropout.coarse_dropout.CoarseDropout)           | ✓     | ✓    |        | ✓         |
| [Crop（裁剪）](https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/#albumentations.augmentations.crops.transforms.Crop)                                         | ✓     | ✓    | ✓      | ✓         |
| [CropAndPad（裁剪和填充）](https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/#albumentations.augmentations.crops.transforms.CropAndPad)                             | ✓     | ✓    | ✓      | ✓         |
| [CropNonEmptyMaskIfExists（如果存在非空掩码则裁剪）](https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/#albumentations.augmentations.crops.transforms.CropNonEmptyMaskIfExists) | ✓     | ✓    | ✓      | ✓         |
| [D4（四重数据增强）](https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.D4)                                     | ✓     | ✓    | ✓      | ✓         |
| [ElasticTransform（弹性变换）](https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.ElasticTransform)         | ✓     | ✓    | ✓      |           |
| [Flip（翻转）](https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.Flip)                                 | ✓     | ✓    | ✓      | ✓         |
| [GridDistortion（网格失真）](https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.GridDistortion)             | ✓     | ✓    | ✓      |           |
| [GridDropout（网格丢失）](https://albumentations.ai/docs/api_reference/augmentations/dropout/grid_dropout/#albumentations.augmentations.dropout.grid_dropout.GridDropout)                   | ✓     | ✓    |        |           |
| [HorizontalFlip（水平翻转）](https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.HorizontalFlip)             | ✓     | ✓    | ✓      | ✓         |
| [Lambda（自定义函数）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Lambda)                                                 | ✓     | ✓    | ✓      | ✓         |
| [LongestMaxSize（最大尺寸调整）](https://albumentations.ai/docs/api_reference/augmentations/geometric/resize/#albumentations.augmentations.geometric.resize.LongestMaxSize)                     | ✓     | ✓    | ✓      | ✓         |
| [MaskDropout（掩码丢失）](https://albumentations.ai/docs/api_reference/augmentations/dropout/mask_dropout/#albumentations.augmentations.dropout.mask_dropout.MaskDropout)                   | ✓     | ✓    |        |           |
| [Morphological（形态学变换）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Morphological)                                   | ✓     | ✓    |        |           |
| [NoOp（无操作）](https://albumentations.ai/docs/api_reference/core/transforms_interface/#albumentations.core.transforms_interface.NoOp)                                                   | ✓     | ✓    | ✓      | ✓         |
| [OpticalDistortion（光学失真）](https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.OpticalDistortion)       | ✓     | ✓    | ✓      |           |
| [PadIfNeeded（根据需要填充）](https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.PadIfNeeded)                   | ✓     | ✓    | ✓      | ✓         |
| [Perspective（透视变换）](https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.Perspective)                   | ✓     | ✓    | ✓      | ✓         |
| [PiecewiseAffine（分段仿射）](https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.PiecewiseAffine)           | ✓     | ✓    | ✓      | ✓         |
| [PixelDropout（像素丢失）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.PixelDropout)                                     | ✓     | ✓    |        |           |
| [RandomCrop（随机裁剪）](https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/#albumentations.augmentations.crops.transforms.RandomCrop)                             | ✓     | ✓    | ✓      | ✓         |
| [RandomCropFromBorders（从边界随机裁剪）](https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/#albumentations.augmentations.crops.transforms.RandomCropFromBorders)       | ✓     | ✓    | ✓      | ✓         |
| [RandomGridShuffle（随机网格洗牌）](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomGridShuffle)                           | ✓     | ✓    |        | ✓         |
| [RandomResizedCrop（随机调整大小的裁剪）](https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/#albumentations.augmentations.crops.transforms.RandomResizedCrop)               | ✓     | ✓    | ✓      | ✓         |
| [RandomRotate90（随机旋转90度）](https://albumentations.ai/docs/api_reference/augmentations/geometric/rotate/#albumentations.augmentations.geometric.rotate.RandomRotate90)                     | ✓     | ✓    | ✓      | ✓         |
| [RandomScale（随机缩放）](https://albumentations.ai/docs/api_reference/augmentations/geometric/resize/#albumentations.augmentations.geometric.resize.RandomScale)                           | ✓     | ✓    | ✓      | ✓         |
| [RandomSizedBBoxSafeCrop（随机尺寸边界框安全裁剪）](https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/#albumentations.augmentations.crops.transforms.RandomSizedBBoxSafeCrop)   | ✓     | ✓    | ✓      | ✓         |
| [RandomSizedCrop（随机尺寸裁剪）](https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/#albumentations.augmentations.crops.transforms.RandomSizedCrop)                   | ✓     | ✓    | ✓      | ✓         |
| [Resize（调整大小）](https://albumentations.ai/docs/api_reference/augmentations/geometric/resize/#albumentations.augmentations.geometric.resize.Resize)                                     | ✓     | ✓    | ✓      | ✓         |
| [Rotate（旋转）](https://albumentations.ai/docs/api_reference/augmentations/geometric/rotate/#albumentations.augmentations.geometric.rotate.Rotate)                                     | ✓     | ✓    | ✓      | ✓         |
| [SafeRotate（安全旋转）](https://albumentations.ai/docs/api_reference/augmentations/geometric/rotate/#albumentations.augmentations.geometric.rotate.SafeRotate)                             | ✓     | ✓    | ✓      | ✓         |
| [ShiftScaleRotate（位移缩放旋转）](https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.ShiftScaleRotate)         | ✓     | ✓    | ✓      | ✓         |
| [SmallestMaxSize（最小最大尺寸）](https://albumentations.ai/docs/api_reference/augmentations/geometric/resize/#albumentations.augmentations.geometric.resize.SmallestMaxSize)                   | ✓     | ✓    | ✓      | ✓         |
| [Transpose（转置）](https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.Transpose)                       | ✓     | ✓    | ✓      | ✓         |
| [VerticalFlip（垂直翻转）](https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.VerticalFlip)                 | ✓     | ✓    | ✓      | ✓         |
| [XYMasking（XY掩码）](https://albumentations.ai/docs/api_reference/augmentations/dropout/xy_masking/#albumentations.augmentations.dropout.xy_masking.XYMasking)                           | ✓     | ✓    |        | ✓         |


### 混合级变换

混合级变换将多个图像合成为一个图像。

| 变换名称                                                                                                                                                       | 图像 | 掩码 | 边界框 | 关键点 | 全局标签 |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---: | :--: | :----: | :-------: | :----------: |
| [MixUp（混合叠加）](https://albumentations.ai/docs/api_reference/augmentations/mixing/transforms/#albumentations.augmentations.mixing.transforms.MixUp)                     | ✓     | ✓    |        |           | ✓            |
| [OverlayElements（叠加元素）](https://albumentations.ai/docs/api_reference/augmentations/mixing/transforms/#albumentations.augmentations.mixing.transforms.OverlayElements) | ✓     | ✓    |        |           |              |

## 更多增强实例

### Inria数据集上的语义分割

![inria](https://habrastorage.org/webt/su/wa/np/suwanpeo6ww7wpwtobtrzd_cg20.jpeg)

### 医学影像

![medical](https://habrastorage.org/webt/1i/fi/wz/1ifiwzy0lxetc4nwjvss-71nkw0.jpeg)

### Mapillary Vistas数据集上的目标检测和语义分割

![vistas](https://habrastorage.org/webt/rz/-h/3j/rz-h3jalbxic8o_fhucxysts4tc.jpeg)

### 关键点增强

<img src="https://habrastorage.org/webt/e-/6k/z-/e-6kz-fugp2heak3jzns3bc-r8o.jpeg" width=100%>

## 基准测试结果

若想自行进行基准测试，请按照 [benchmark/README.md](https://github.com/albumentations-team/albumentations/blob/master/benchmark/README.md) 中的说明操作。

使用AMD Ryzen Threadripper 3970X CPU对ImageNet验证集的前2000张图片进行基准测试的结果。表格显示了单个核心每秒可以处理的图像数量；数值越大越好。

| 库 | 版本 |
|---------|---------|
| Python | 3.10.13 (主版本, 2023年9月11日, 13:44:35) [GCC 11.2.0] |
| albumentations | 1.4.11 |
| imgaug | 0.4.0 |
| torchvision | 0.18.1+rocm6.0 |
| numpy | 1.26.4 |
| opencv-python-headless | 4.10.0.84 |
| scikit-image | 0.24.0 |
| scipy | 1.14.0 |
| pillow | 10.4.0 |
| kornia | 0.7.3 |
| augly | 1.0.0 |

|                 |albumentations<br><small>1.4.11</small>|torchvision<br><small>0.18.1+rocm6.0</small>|kornia<br><small>0.7.3</small>|augly<br><small>1.0.0</small>|imgaug<br><small>0.4.0</small>|
|-----------------|--------------------------------------|--------------------------------------------|------------------------------|-----------------------------|------------------------------|
|HorizontalFlip   |**8017 ± 12**                         |2436 ± 2                                    |935 ± 3                       |3575 ± 4                     |4806 ± 7                      |
|VerticalFlip     |7366 ± 7                              |2563 ± 8                                    |943 ± 1                       |4949 ± 5                     |**8159 ± 21**                 |
|Rotate           |570 ± 12                              |152 ± 2                                     |207 ± 1                       |**633 ± 2**                  |496 ± 2                       |
|Affine           |**1382 ± 31**                         |162 ± 1                                     |201 ± 1                       |-                            |682 ± 2                       |
|Equalize         |1027 ± 2                              |336 ± 2                                     |77 ± 1                        |-                            |**1183 ± 1**                  |
|RandomCrop64     |**19986 ± 57**                        |15336 ± 16                                  |811 ± 1                       |19882 ± 356                  |5410 ± 5                      |
|RandomResizedCrop|**2308 ± 7**                          |1046 ± 3                                    |187 ± 1                       |-                            |-                             |
|ShiftRGB         |1240 ± 3                              |-                                           |425 ± 2                       |-                            |**1554 ± 6**                  |
|Resize           |**2314 ± 9**                          |1272 ± 3                                    |201 ± 3                       |431 ± 1                      |1715 ± 2                      |
|RandomGamma      |**2552 ± 2**                          |232 ± 1                                     |211 ± 1                       |-                            |1794 ± 1                      |
|Grayscale        |**7313 ± 4**                          |1652 ± 2                                    |443 ± 2                       |2639 ± 2                     |1171 ± 23                     |
|ColorJitter      |**396 ± 1**                           |51 ± 1                                      |50 ± 1                        |224 ± 1                      |-                             |
|PlankianJitter   |449 ± 1                               |-                                           |**598 ± 1**                   |-                            |-                             |
|RandomPerspective|471 ± 1                               |123 ± 1                                     |114 ± 1                       |-                            |**478 ± 2**                   |
|GaussianBlur     |**2099 ± 2**                          |113 ± 2                                     |79 ± 2                        |165 ± 1                      |1244 ± 2                      |
|MedianBlur       |538 ± 1                               |-                                           |3 ± 1                         |-                            |**565 ± 1**                   |
|MotionBlur       |**2197 ± 9**                          |-                                           |102 ± 1                       |-                            |508 ± 1                       |
|Posterize        |2449 ± 1                              |**2587 ± 3**                                |339 ± 6                       |-                            |1547 ± 1                      |
|JpegCompression  |**827 ± 1**                           |-                                           |50 ± 2                        |684 ± 1                      |428 ± 4                       |
|GaussianNoise    |78 ± 1                                |-                                           |-                             |67 ± 1                       |**128 ± 1**                   |
|Elastic          |127 ± 1                               |3 ± 1                                       |1 ± 1                         |-                            |**130 ± 1**                   |
|Normalize        |**971 ± 2**                           |449 ± 1                                     |415 ± 1                       |-                            |-                             |

## 参与贡献

若想对仓库创建拉取请求，请遵循[CONTRIBUTING.md](CONTRIBUTING.md)中的文档。

![https://github.com/albuemntations-team/albumentation/graphs/contributors](https://contrib.rocks/image?repo=albumentations-team/albumentations)

## 社区与支持

- [Twitter](https://twitter.com/albumentations)
- [Discord](https://discord.gg/AKPrrDYNAt)

## 评论

在某些系统中，如果OpenCV编译了OpenCL优化，在多GPU模式下，PyTorch可能会使DataLoader死锁。在导入库之前添加以下两行代码可能会有帮助。更多详情请访问[https://github.com/pytorch/pytorch/issues/1355](https://github.com/pytorch/pytorch/issues/1355)

```python
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
```

## 引用

如果您认为这个库对您的研究有用，请考虑引用[Albumentations: Fast and Flexible Image Augmentations](https://www.mdpi.com/2078-2489/11/2/125):

```bibtex
@Article{info11020125,
    AUTHOR = {Buslaev, Alexander and Iglovikov, Vladimir I. and Khvedchenya, Eugene and Parinov, Alex and Druzhinin, Mikhail and Kalinin, Alexandr A.},
    TITLE = {Albumentations: Fast and Flexible Image Augmentations},
    JOURNAL = {Information},
    VOLUME = {11},
    YEAR = {2020},
    NUMBER = {2},
    ARTICLE-NUMBER = {125},
    URL = {https://www.mdpi.com/2078-2489/11/2/125},
    ISSN = {2078-2489},
    DOI = {10.3390/info11020125}
}
```