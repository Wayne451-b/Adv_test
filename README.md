## Defending Deepfakes by Attribute-Aware Attack

### Files
1. ./checkpoints - 训练模型保存目录
2. ./networks - 模型文件
3. ./test_samples - 测试图片目录
4. ./tools - 模块工具文件
config.py - 配置文件
data.py - CelebA-HQ数据预处理脚本
logger.py - 日志文件
test.py - CelebA-HQ数据集测试脚本
infer.py - 自定义单图推理脚本
train.py - 训练脚本

### Dataset
下载[CelebA-HQ数据集](https://github.com/switchablenorms/CelebAMask-HQ)，通过--dataset_path与--attribute_txt_path参数传入数据集路径和属性标签路径

### Inference
下载模型预训练权重，保存至./checkpoints/目录下，详见./checkpoints/readme.md

运行推理脚本：
```python inference.py```
推理结果保存在demo_results目录下

### Test
运行测试脚本：
```python test.py```
测试结果保存在test_results目录下

### Train
运行训练脚本：
```python train.py```
参考 Proactive-Defense-Against-Facial-Editing-in-DWT 的跨模型集成防御训练方法

### Acknowledgement
非常感谢以下项目给予的贡献与帮助：
This code is based on [Proactive-Defense-Against-Facial-Editing's benchmark](https://github.com/imagecbj/Proactive-Defense-Against-Facial-Editing-in-DWT), [BiSeNet](https://github.com/yakhyo/face-parsing), [AdvGAN](https://github.com/mathcbc/advGAN_pytorch) and [UVCGAN](https://github.com/LS4GAN/uvcgan) etc.
Thanks for their great contribution.