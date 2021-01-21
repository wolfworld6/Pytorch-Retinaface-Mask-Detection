Pytorch版本Retinaface, 原始代码为[Pytorch_Retinface](https://github.com/biubug6/Pytorch_Retinaface/tree/master/models). 修改以支持口罩检测以及自定义数据格式，并添加一些有用的脚本.



## 安装
1. git clone https://github.com/upczww/Pytorch-retinaface-mask-detection.git

## 环境
Pytorch 1.1.0 和 torchvision 0.3.0以上版本

## 准备数据
按以下形式组织数据
```
  ./data/kouzhao/
    train/
      images/
      labels.txt
    val/
      images/
      labels.txt
```
`labels.txt`的每一行为一张图片对应的标签，格式如下
```
image1.jpg\t161,97,277,253,2\space86,89,159,170,1\n
```
分别表示图片路径，标注框和标签(x_min,y_min,x_max,y_max,label), 1 表示人脸无口罩, 2 表示人脸有口罩. 第一个分隔符为\t, 后面为空格.

## 训练
1. 修改配置文件 `data/config.py` 和 `train.py`, 比如 batch_size, 学习率, 输入大小等.

2. 训练脚本
  ```
  CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --network resnet50 
  CUDA_VISIBLE_DEVICES=0 python train.py --network mobile0.25
  ```
## 测试
1. 生成测试结果
```
CUDA_VISIBLE_DEVICES=0 python test.py -m weights/Resnet50_Final.pth -s
```
将会在 `input/detection-result-all`中保存结果.

2. 使用不同的分数过滤结果，比如使用 0.5.
```
python scripts/filter_results.py -t 0.5
```

将会在 `input/detection-result`中保存结果.

3. 生成 ground truth
```
python scripts/generate_ground_truth.py -s data/kouzhao/test/labels.txt -d input/ground-truth
```
4. 计算 mAP, 使用不同的iou阈值
```
python script/cal_mAP.py -o 0.5
```
5. 测试单张图片
```
python test_one.py --help
```

### 导出权重 .wts
```
python script/genwts --help
```