import importlib
import json
import math
import os
import time

import librosa
import numpy as np
import torch

MEL_FILTER = librosa.filters.mel(16000, 1024, n_mels=256)

class ExecutionTime:
    """
    Usage:
        timer = ExecutionTime()
        <Something...>
        print(f'Finished in {timer.duration()} seconds.')
    """


    def __init__(self):
        self.start_time = time.time()


    def duration(self):
        return time.time() - self.start_time


def find_aligned_wav_files(dir_a, dir_b, limit=0, offset=0):
    """
    搜索 dir_A 与 dir_B 根目录下的 wav 文件，要求：
        - 两个目录中的 wav 文件数量相等
        - 索引相同，文件名也相同，排序方式按照 Python 內建的 .sort() 函数
    Args:
        dir_a:  目录 A
        dir_b: 目录 B
        limit: 加载 wav 文件的数量限制
        offset: 开始位置的偏移索引

    Notes:
        length:
            1. limit == 0 and limit > len(wav_paths_in_dir_a) 时，length 为 目录下所有文件
            2. limit <= len(wav_paths_in_dir_a) 时，length = limit
    """

    if limit == 0:
        # 当 limit == None 时，librosa 会加载全部文件
        limit = None

    wav_paths_in_dir_a = librosa.util.find_files(dir_a, ext="wav", limit=limit, offset=offset)
    wav_paths_in_dir_b = librosa.util.find_files(dir_b, ext="wav", limit=limit, offset=offset)

    length = len(wav_paths_in_dir_a)

    # 两个目录数量相等，且文件数量 > 0
    assert len(wav_paths_in_dir_a) == len(wav_paths_in_dir_b) > 0, f"目录 {dir_a} 和目录 {dir_b} 文件数量不同或目录为空"

    # 两个目录中的 wav 文件应当文件名一一对应
    for wav_path_a, wav_path_b in zip(wav_paths_in_dir_a, wav_paths_in_dir_b):
        assert os.path.basename(wav_path_a) == os.path.basename(wav_path_b), \
            f"{wav_path_a} 与 {wav_path_a} 不对称，这可能由于两个目录文件数量不同"

    return wav_paths_in_dir_a, wav_paths_in_dir_b, length


def set_requires_grad(nets, requires_grad=False):
    """
    修改多个网络梯度
    Args:
        nets: list of networks
        requires_grad: 是否需要梯度
    """

    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def sample_fixed_length_data_aligned(data_a, data_b, sample_length):
    """
    对 data_a 与 data_b 进行对齐采样
    Args:
        data_a:
        data_b:
        sample_length: 采样的点数
    """
    assert len(data_a) == len(data_b), "数据长度不一致，无法完成定长采样"
    assert len(data_a) >= sample_length, f"len(data_a) is {len(data_a)}, sample_length is {sample_length}."

    frames_total = len(data_a)

    start = np.random.randint(frames_total - sample_length + 1)
    # print(f"Random crop from: {start}")
    end = start + sample_length

    return data_a[start:end], data_b[start:end]


def sample_dataset_aligned(dataset_A, dataset_B, n_frames=128):
    """
    将变长的数据样本采样为定长的数据样本
    Args:
        dataset_A: 数据集 A，内部元素为变长，比如为 [[128, 325], [128, 356], ...]
        dataset_B: 数据集 B，内部元素为变长, 比如为 [[128, 325], [128, 356], ...]
        n_frames: 采样的帧数，默认为 128
    Returns:
        采样后的数据集 A，内部元素为定长: [[128, 128], [128, 128], ...]
        采样后的数据集 B，内部元素为定长: [[128, 128], [128, 128], ...]
    """

    data_A_idx = np.arange(len(dataset_A))
    data_B_idx = np.arange(len(dataset_B))

    sampling_dataset_A = list()
    sampling_dataset_B = list()

    for idx_A, idx_B in zip(data_A_idx, data_B_idx):
        # 获取样本
        data_A = dataset_A[idx_A]
        data_B = dataset_B[idx_B]

        # 样本中的帧数
        frames_A_total = data_A.shape[1]
        frames_B_total = data_B.shape[1]
        assert frames_A_total == frames_B_total, "A 样本和 B 样本的帧数不同，样本的索引为 {}.".format(idx_A)

        # 确定采样的起止位置，将变长样本采样为定长样本
        assert frames_A_total >= n_frames
        start = np.random.randint(frames_A_total - n_frames + 1)
        end = start + n_frames
        sampling_dataset_A.append(data_A[:, start: end])
        sampling_dataset_B.append(data_B[:, start: end])

    sampling_dataset_A = np.array(sampling_dataset_A)
    sampling_dataset_B = np.array(sampling_dataset_B)

    return sampling_dataset_A, sampling_dataset_B

def calculate_l_out(l_in, kernel_size, stride, dilation=1, padding=0):
    # https://pytorch.org/docs/stable/nn.html#conv1d
    return math.floor(((l_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)

def calculate_same_padding(l_in, kernel_size, stride, dilation=1):
    # https://pytorch.org/docs/stable/nn.html#conv1d
    return math.ceil(((l_in - 1) * stride + 1 + dilation * (kernel_size - 1) - l_in) / 2)

def initialize_config_in_single_module(module_cfg, module):
    """
    根据配置项，获取模块内部的函数，并将参数传入函数
    Args:
        module_cfg (dict): 针对 module 模块的配置信息
        module: 调用其内部的属性（函数）

    Returns:
        调用模块内对应的函数，返回函数执行后的返回值
    """
    # 调用模块内对应的函数，返回函数执行后的返回值
    return getattr(module, module_cfg["type"])(**module_cfg["args"])

def initialize_config(module_cfg):
    """
    根据配置项，动态加载对应的模块，并将参数传入模块内部的指定函数
    eg，配置文件如下：
        module_cfg = {
            "module": "models.unet",
            "main": "UNet",
            "args": {...}
        }
    1. 加载 type 参数对应的模块
    2. 调用（实例化）模块内部对应 main 参数的函数（类）
    3. 在调用（实例化）时将 args 参数输入函数（类）

    Args:
        module_cfg: 配置信息，见 json 配置文件

    Returns:

    """
    module = importlib.import_module(module_cfg["module"])
    return getattr(module, module_cfg["main"])(**module_cfg["args"])

def write_json(content, path):
    with open(path, "w") as handle:
        json.dump(content, handle, indent=2, sort_keys=False)

def apply_mean_std(y):
    return (y - np.mean(y)) / np.std(y)



def stft(y, n_fft=512, hop_length=256, win_length=512, window="hamming"):
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
    mag, phase = librosa.magphase(D, power=1)
    return mag, phase

# def istft(mag, phase, )
#     D = mag * noisy_phase
#     return librosa.istft(mag * noisy_phase, hop_length=256, win_length=512, window='hamming')

def mag(y):
    D = librosa.stft(y, n_fft=512, hop_length=256, window='hamming')
    return np.abs(D)


def input_normalization(m):
    mean = np.mean(m)
    std_var = np.std(m)
    return (m - mean) / std_var

def unfold_spectrum(spec, n_pad=3):
    """
    对频谱应用滑窗操作

    Args:
        spec (np.array): 频谱，(n_fft, T)
        n_pad (int): 输入帧 pad 的大小 (default: 3，即左边 3 帧，右边也是 3 帧)

    Returns:
        np.array -- 拓展过频谱，尺寸为 (n_fft, T * (n_pad * 2 + 1))
    """
    # 补齐频谱左侧后右侧
    left_pad_spec = np.repeat(spec[:, 0].reshape(-1, 1), n_pad, axis=1)  # (257, 3)
    right_pad_spec = np.repeat(spec[:, -1].reshape(-1, 1), n_pad, axis=1)  # (257, 3)
    assert left_pad_spec.shape[-1] == right_pad_spec.shape[-1] == n_pad
    spec = np.concatenate([left_pad_spec, spec, right_pad_spec], axis=1).T  # (120, 257)
    spec = torch.Tensor(spec)

    # 类似于滑窗的效果，窗大小为 2*n_pad+1，每次滑动的间隔为 1
    spec_list = spec.unfold(0, 2 * n_pad + 1, 1)  # [tensor(257, 7), tensor(257, 7), ...], len = 114
    spec = torch.cat(tuple(spec_list), dim=1).numpy()  # (257, 798)

    return spec

def cal_lps(y, pad=0):
    D = librosa.stft(y, n_fft=512, hop_length=256, window='hamming')
    mag = np.abs(D)
    lps = np.log(np.power(mag, 2))
    if pad != 0:
        lps = np.concatenate((np.zeros((257, pad)), lps, np.zeros((257, pad))), axis=1)
    return lps

def cal_mag(y):
    D = librosa.stft(y, n_fft=512, hop_length=256, window='hamming')
    return np.abs(D)

def cal_mel(y):
    """
    Notes:
        若使用 librosa.filters.cal_mel(16000, 512, n_mels=256) 会出现 cal_mel 频率不够分的情况
        本函数将 n_fft 修改为 1024，与 lps，mag 函数中的设置不同
    Returns:
        cal_mel: (256, T)
    """
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256, window="hamming")) ** 2
    return np.dot(MEL_FILTER, S)

def rebuild_waveform(mag, noisy_phase):
    return librosa.istft(mag * noisy_phase, hop_length=256, win_length=512, window='hamming')

def phase(y, n_fft=512):
    D = librosa.stft(y, n_fft=n_fft, hop_length=256, window='hamming')
    _, phase = librosa.magphase(D)
    return phase

def lps_to_mag(lps):
    return np.power(np.exp(lps), 1 / 2)

def cal_log_mel(y):
    return cal_mel(y) ** 0.125

def log_mel_to_mag(log_mel):
    mel = np.power(log_mel, 1 / 0.125)
    return mel_to_mag(mel)

def mel_to_mag(mel):
    mag = np.dot(MEL_FILTER.T, mel)
    mag = np.power(mag, 1 / 2)
    return mag