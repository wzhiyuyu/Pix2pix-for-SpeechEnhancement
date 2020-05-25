# Pix2pix for Speech Enhancement

使用 cGAN 进行语音增强的项目

## Features

- 增加了层数的 Pix2pix
- 整合了语音预处理与重建的相关代码
- 包含丰富的数据可视化

## Requirements
- Python >= 3.6
- PyTorch >= 1.1
- tqdm
- tensorboard
- python-pesq
- matplotli
- pystoi
- joblib

## Folder Structure

```shell
├── config
│   ├── dev.yaml # 每次独立的实验都需要新建一个这样的文件，用于覆盖默认的初始配置项
│   ├── test_config.py # 测试时的初始配置项
│   └── train_config.py # 训练时的初始配置项
├── data
│   ├── __init__.py
│   ├── test_dataset.py # 测试数据集，加载结果为语音信号
│   └── train_dataset.py # 训练数据集，加载结果为 Mel 频谱
├── models
│   ├── __init__.py
│   ├── loss.py # 定义模型的损失函数
│   ├── metric.py # 定义评价指标
│   ├── module.py # 模型组件
│   ├── pix2pix_model.py
│   ├── gated_residual_model.py
├── trainer
│   ├── __init__.py
│   ├── base_trainer.py # 基类：初始化参数，新建空目录，配置 GPU 设备，保存与加载模型断点
│   └── trainer.py # 训练：定义单次训练，单次验证，单次测试的逻辑，计算评价指标，定义训练总循环
├── README.md
├── requirements.txt
├── test.py
├── train.py
└── utils
    ├── __init__.py
    ├── util.py
    └── visualization.py # 可视化相关
```


## Usage

### Train

Try `python train.py -C <config file> -D 0` to run training code. 

```shell
python train.py -C <config file> -D <gpu ids> [-R]

-C CONFIG, --config CONFIG
                    包含训练参数的 json 文件
-D DEVICE, --device DEVICE
                    当前训练过程可见的 GPU 索引，比如：0,1,2,3
-R, --resume          
                    从最近的一个断点处继续训练
```

#### Training config file

```json
{   
    "_comment": "training session name",
    "name": "pix2pix",
    "_comment": "training session description",
    "description": "pix2pix",
    "_comment": "location to saving training session",
    "save_location": "/media/imucs/DataDisk/wangzhiyu/Experiment/Pix2pix",
    "_comment": "number of GPUs to use for training.",
    "n_gpu": 1,
    "_comment": "using cudnn",
    "use_cudnn": true,
    "model": {
        "module": "models.unet_dilation",
        "main": "UNet",
        "args": {}
    },
    "loss_function": {
        "module": "models.loss",
        "main": "mse_loss",
        "args": {}
    },
    "optimizer": {
        "lr": 0.001
    },
    "trainer": {
        "epochs": 1000,
        "save_period": 10,
        "validation_period": 5
    },
    "train_dataset": {
        "mixture": "/media/imucs/DataDisk/wangzhiyu/Release/speech_enhancement/noise_7_clean_900/train/mixture.npy",
        "clean": "/media/imucs/DataDisk/wangzhiyu/Release/speech_enhancement/noise_7_clean_900/train/clean.npy",
        "limit": null,
        "offset": 0,
        "shuffle": true,
        "num_workers": 40,
        "batch_size": 150
    },
    "valid_dataset": {
        "mixture": "/media/imucs/DataDisk/wangzhiyu/Release/speech_enhancement/noise_7_clean_900/test/mixture.npy",
        "clean": "/media/imucs/DataDisk/wangzhiyu/Release/speech_enhancement/noise_7_clean_900/test/clean.npy",
        "limit": 100,
        "offset": 0
    }
}
```

### Enhancement

ToDo

### 可视化

Training session 会存储至训练配置文件的 `config[save_location][name]` 中，包含以下内容：

- `logs/`: `Tensorboard` 相关的数据，包含波形文件，语音文件，损失曲线等
- `checkpoints/`: 模型断点，用于后续的恢复训练
- `config.json`: 训练相关的配置信息

使用 `tensorboardX` 可以可视化数据:

```shell
tensorboard --logdir config[save_location]/[name]/logs

# 可使用 --port 指定 tensorboardX 服务器的端口
tensorboard --logdir config[save_location]/[name]/logs --port <port>
```

在 `http://localhost:6006` 中查看数据。

### Notes:

- 不能在模型保存时暂停模型，这会使断点存储的就会不完整
- `.to()`，`cpu()` 和 `cuda()` 会直接将模型放在对应的设备上，再转换要及时转回来
- 合理的 validate_limit 可以轻松的找到 STOI 与 PESQ 表现最好的模型。但 validate_limit 参数也会影响 tensorboardX 可视化页面的流畅度、计算速度

## ToDo

- [ ] 

## License

This project is licensed under the MIT License.
