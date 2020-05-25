import argparse
import json
from pathlib import Path

import numpy as np
import tablib
import torch
from joblib import Parallel, delayed
from torch.utils.data import DataLoader

from utils.metric import compute_STOI, compute_PESQ
from utils.utils import initialize_config, cal_mel, cal_log_mel, cal_mag, cal_lps, phase, rebuild_waveform, mel_to_mag, \
    log_mel_to_mag, lps_to_mag


def extract_spectrum(y, spectrum_type="mel"):
    last_bin_spectrum = None

    if spectrum_type == "mel":
        spectrum = cal_mel(y)
    elif spectrum_type == "log_mel":
        spectrum = cal_log_mel(y)
    elif spectrum_type == "lps":
        spectrum = cal_lps(y)
    elif spectrum_type == "mag":
        spectrum = cal_mag(y)
    else:
        raise ValueError("特征类型出错，应该为 [mel, log_mel, mag, lps] 中的一种")

    if spectrum_type in ("lps", "mag"):
        spectrum = spectrum[:, :-1]  # 抛弃最后一帧
        last_bin_spectrum = spectrum[-1, :]
        spectrum = spectrum[:-1, :]  # 原尺寸为 (257, T)，为了方便计算，抛弃最后一个 bin

    return spectrum, last_bin_spectrum


def extract_phase_spectrum(y, spectrum_type="mel"):
    if spectrum_type in ("mel", "log_mel"):
        phase_spectrum = phase(y, n_fft=1024)
    else:
        phase_spectrum = phase(y)

    if spectrum_type in ("lps", "mag"):
        # 其实没有必要丢掉最后一帧
        phase_spectrum = phase_spectrum[:, :-1]

    return phase_spectrum

def spectrum_to_waveform(spectrum, phase_spectrum, last_bin, spectrum_type):
    if spectrum_type == "mag":
        # 带噪语音的 last_bin 被用在增强后的语音中，但需要先删除额外的长度
        last_bin = last_bin[:spectrum_type.shape[1]].reshape(1, -1)
        spectrum = np.concatenate([spectrum, last_bin], axis=0)  # Pad 最后一个 bin
        assert spectrum.shape[0] == 257
    elif spectrum_type == "lps":
        # 带噪语音的 last_bin 被用在增强后的语音中，但需要先删除额外的长度
        last_bin = last_bin[:spectrum_type.shape[1]].reshape(1, -1)
        spectrum = np.concatenate([spectrum, last_bin], axis=0)  # Pad 最后一个 bin
        assert spectrum.shape[0] == 257
        spectrum = lps_to_mag(spectrum)
    elif spectrum_type == "mel":
        spectrum = mel_to_mag(spectrum)
    else:
        spectrum = log_mel_to_mag(spectrum)

    waveform = rebuild_waveform(spectrum, phase_spectrum)
    return waveform

def enhance_spectrum(model, mixture, clean, mixture_name, norm_meta, spectrum_type="mel"):
    mixture_spectrum, last_bin = extract_spectrum(mixture, spectrum_type)
    mixture_phase = extract_phase_spectrum(mixture, spectrum_type)

    mixture_spectrum_chunks = torch.split(torch.tensor(mixture_spectrum), 256, dim=1)
    if len(mixture_spectrum_chunks) > 1:
        # 在构建数据集时已经保证了可以提取出 T >= 256 的频谱
        # 当 t 正好等于 256 时，不能再丢掉最后一个了，a = [1], a[:-1] == []
        mixture_spectrum_chunks = mixture_spectrum_chunks[:-1]

    # 增强频谱
    enhanced_spectrum_chunks = []
    for i, chunk in enumerate(mixture_spectrum_chunks):
        norm_max = torch.max(chunk)
        norm_min = torch.min(chunk)
        chunk = 2 * (chunk - norm_min) / (norm_max - norm_min) - 1
        chunk = chunk.reshape(1, 1, 256, 256)
        enhanced_chunk = model(chunk).squeeze()  # [256, 256]
        enhanced_chunk = (enhanced_chunk + 1) * (norm_max - norm_min) / 2 + norm_min
        enhanced_spectrum_chunks.append(enhanced_chunk.detach().numpy())

    enhanced_spectrum = np.concatenate(enhanced_spectrum_chunks, axis=1)
    assert enhanced_spectrum.shape[0] == 256  # (256, T)

    enhanced = spectrum_to_waveform(enhanced_spectrum, mixture_phase[:, :enhanced_spectrum.shape[1]], last_bin,
                                    spectrum_type)
    min_len = min(len(mixture), len(clean), len(enhanced))
    mixture = mixture[:min_len]
    enhanced = enhanced[:min_len]
    clean = clean[:min_len]

    stoi_c_m = compute_STOI(clean, mixture, sr=16000)
    stoi_c_e = compute_STOI(clean, enhanced, sr=16000)
    pesq_c_m = compute_PESQ(clean, mixture, sr=16000)
    pesq_c_e = compute_PESQ(clean, enhanced, sr=16000)

    num, noise, snr = mixture_name.split("_")

    print((
        num, noise, snr,
        stoi_c_m, stoi_c_e,
        pesq_c_m, pesq_c_e,
        (stoi_c_e - stoi_c_m) / stoi_c_m,
        (pesq_c_e - pesq_c_m) / pesq_c_m
    ))

    return (
        num, noise, snr,
        stoi_c_m, stoi_c_e,
        pesq_c_m, pesq_c_e,
        (stoi_c_e - stoi_c_m) / stoi_c_m,
        (pesq_c_e - pesq_c_m) / pesq_c_m
    )

@torch.no_grad()
def main(config, epoch):
    root_dir = Path(config["save_location"]) / config["name"]
    checkpoints_dir = root_dir / "checkpoints"

    """============== 加载数据集 =============="""
    dataset = initialize_config(config["dataset"])
    collate_fn = lambda data_list: data_list
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=40,
        num_workers=40,
        collate_fn=collate_fn,
        drop_last=True
    )

    """============== 加载模型断点（best，latest，通过数字指定） =============="""
    model = initialize_config(config["model"])
    device = torch.device("cpu")
    model = model.double()

    if epoch == "best":
        model_path = checkpoints_dir / "best_model.tar"
    elif epoch == "latest":
        model_path = checkpoints_dir / "latest_model.tar"
    else:
        model_path = checkpoints_dir / f"generator_{str(epoch).zfill(3)}.pth"

    model_checkpoint = torch.load(model_path.as_posix(), map_location=device)
    model_static_dict = model_checkpoint["model_state_dict_G"]
    print(f"Loading model checkpoint, epoch = {model_checkpoint['epoch']}")
    model.load_state_dict(model_static_dict)
    model.eval()

    """============== 增强语音 =============="""
    results_dir = root_dir / f"epoch_{epoch}_results"
    results_dir.mkdir(parents=False, exist_ok=True)
    spectrum_type = config["spectrum_type"]

    headers = ("语音编号", "噪声类型", "信噪比",
               "STOI 纯净与带噪", "STOI 纯净与降噪 ",
               "PESQ 纯净与带噪", "PESQ 纯净与降噪",
               "STOI 提升",
               "PESQ 提升")  # 定义导出为 Excel 文件的格式
    metrics_seq = []

    for i, data in enumerate(dataloader):
        # batch_size个结果被汇总在一个 list 中了
        # [(return_1,return_2, ... ,return_40), ..., ]
        metrics = Parallel(n_jobs=40, require="sharedmem")(delayed(enhance_spectrum)(
            model,
            *data[j],
            spectrum_type
        ) for j in range(dataloader.batch_size))

        metrics_seq += metrics

    """============== 存储结果 =============="""
    data = tablib.Dataset(*metrics_seq, headers=headers)
    metrics_save_dir = results_dir / f"epoch_{epoch}.xls"
    with open(metrics_save_dir.as_posix(), 'wb') as f:
        f.write(data.export('xls'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pix2pix for Speech Enhancement")
    parser.add_argument("-C", "--config", default="config/test/config.json", type=str,
                        help="测试脚本的配置项。(default: config/test/config.json)")
    parser.add_argument("-E", "--epoch", default="best", help="加载断点的轮次，可以设置为 best, latest, 具体轮次。(default: best)")
    args = parser.parse_args()

    config = json.load(open(args.config))
    main(config, args.epoch)
