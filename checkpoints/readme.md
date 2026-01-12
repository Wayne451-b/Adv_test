### 预训练权重文件说明
包括代理模型预训练权重、非集成单模型防御预训练权重、跨模型集成防御预训练权重

### 代理模型权重
可在Proactive-Defense-Against-Facial-Editing-in-DWT提供的权重下载地址中找到代理模型的权重：https://drive.google.com/drive/folders/10WEoO6C6KkcFqtVb_iCciDEUqjLPJJEJ

分别下载至
>./checkpoints/attentiongan
>
>./checkpoints/AttGAN
> 
>./checkpoints/FAN
>
>./checkpoints/FGAN
>
>./checkpoints/HiSD

在--face_pretrained_weights与--hair_pretrained_weights中修改预训练权重路径

### BiseNet权重
可在https://github.com/yakhyo/face-parsing中下载resnet34.pt
可在https://github.com/italojs/facial-landmarks-recognition下载到shape_predictor_68_face_landmarks.dat

### 非集成单模型防御预训练权重
google drive中下载：```https://drive.google.com/drive/folders/1bVvmnh6x2N1wbyCAmmgIRjSTgYPeh4Qy?usp=drive_link```
分别下载至
>./checkpoints/single_model_adv/attgan_adv
>
>./checkpoints/single_model_adv/hisd_adv
>
>./checkpoints/single_model_adv/fgan_adv

非集成防御推理时eposilon建议0.01~0.02
在--face_pretrained_weights与--hair_pretrained_weights中修改预训练权重路径

### 跨模型集成防御预训练权重
google drive中下载：```https://drive.google.com/drive/folders/1_YI5hyfHpWVF6YfFxxFVCtr7Rwd75l_B?usp=drive_link```
下载至```./checkpoints/cross_model_adv```
在```--face_pretrained_weights```与```--hair_pretrained_weights```中修改预训练权重路径

其中```face&hair_perturb_mask_out```为PG后加入属性掩码，权重文件选择```./checkpoints/cross_model_adv/face&hair_perturb_mask_out.pth```
通过```python test.py```运行测试脚本，模型选择```PG_face&hair_model```

face&hair_perturb_dualmask为PG前后都加入属性掩码，权重文件选择```./checkpoints/cross_model_adv/face&hair_perturb_dualmask.pth```
通过```python test.py```运行测试脚本，模型选择```PG_face&hair_dualmask```

跨模型集成防御推理时eposilon设置为0.02

