import os
import re
import json
from glob import glob
from typing import List, Tuple, Dict
from datetime import datetime

import numpy as np
import soundfile as sf
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# 基础配置
# -----------------------------

SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
MAX_DURATION = 10.0  # 秒，超过则截断，不足则补零


def get_device(prefer: str = "cuda") -> torch.device:
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def infer_machine_name_from_data_root(data_root: str) -> str:
    """
    根据 data_root 推断设备名称：
    例如 data_root='data/valve' -> 'valve'；data_root='D:/xxx/data/fan/' -> 'fan'
    """
    return os.path.basename(os.path.normpath(data_root))


# -----------------------------
# 特征提取：log-mel 频谱
# -----------------------------

def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    y, file_sr = sf.read(path)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if file_sr != sr:
        # 使用简单线性插值进行重采样，避免依赖 scipy/librosa
        duration = len(y) / float(file_sr)
        old_t = np.linspace(0, duration, num=len(y), endpoint=False)
        new_length = int(duration * sr)
        new_t = np.linspace(0, duration, num=new_length, endpoint=False)
        y = np.interp(new_t, old_t, y).astype(np.float32)

    max_len = int(MAX_DURATION * sr)
    if len(y) > max_len:
        y = y[:max_len]
    elif len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)))
    return y.astype(np.float32)


_mel_filter_cache: Dict[Tuple[int, int, int], np.ndarray] = {}


def _mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    """自行实现 mel 滤波器组，避免依赖 librosa/scipy。"""
    if fmax is None:
        fmax = sr / 2

    key = (sr, n_fft, n_mels)
    if key in _mel_filter_cache:
        return _mel_filter_cache[key]

    def hz_to_mel(f: float) -> float:
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    n_freqs = n_fft // 2 + 1
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mels = np.linspace(mel_min, mel_max, num=n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft * hz / sr)).astype(int)
    bins = np.clip(bins, 0, n_freqs - 1)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_m_minus = bins[m - 1]
        f_m = bins[m]
        f_m_plus = bins[m + 1]
        if f_m_minus == f_m:
            f_m = min(f_m + 1, n_freqs - 1)
        if f_m == f_m_plus:
            f_m_plus = min(f_m_plus + 1, n_freqs - 1)
        for k in range(f_m_minus, f_m):
            fb[m - 1, k] = (k - f_m_minus) / max(f_m - f_m_minus, 1)
        for k in range(f_m, f_m_plus):
            fb[m - 1, k] = (f_m_plus - k) / max(f_m_plus - f_m, 1)

    _mel_filter_cache[key] = fb
    return fb


def wav_to_logmel(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    # 使用 torch.stft 计算功率谱，再乘以 mel 滤波器
    y_t = torch.from_numpy(y.astype(np.float32))
    window = torch.hann_window(N_FFT)
    spec = torch.stft(
        y_t,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window=window,
        center=True,
        return_complex=True,
    )  # [freq, time]
    power = (spec.abs() ** 2).cpu().numpy()  # [freq, time]

    mel_fb = _mel_filterbank(sr=sr, n_fft=N_FFT, n_mels=N_MELS)  # [n_mels, freq]
    mel = np.dot(mel_fb, power)  # [n_mels, time]
    mel = np.maximum(mel, 1e-10)

    # 对数刻度（以 dB 为近似）：log10
    logmel = np.log10(mel)

    # 归一化到 0 均值 / 1 方差，有利于训练
    mean = np.mean(logmel)
    std = np.std(logmel) + 1e-6
    logmel = (logmel - mean) / std
    return logmel.astype(np.float32)


def extract_feature(path: str) -> np.ndarray:
    """用于 CNN section 分类的 2D 特征：log-mel 频谱图。"""
    y = load_audio(path)
    feat = wav_to_logmel(y)
    # [n_mels, time] -> [1, n_mels, time]
    return np.expand_dims(feat, axis=0)


# -----------------------------
# 额外的手工特征（用于异常检测等）
# -----------------------------


def _compute_spectrogram(y: np.ndarray, sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, np.ndarray]:
    """返回幅度谱 [freq, time] 和频率坐标 freqs[freq]。"""
    y_t = torch.from_numpy(y.astype(np.float32))
    window = torch.hann_window(N_FFT)
    spec = torch.stft(
        y_t,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window=window,
        center=True,
        return_complex=True,
    )
    mag = spec.abs().cpu().numpy()  # [freq, time]
    freqs = np.linspace(0, sr / 2.0, mag.shape[0])
    return mag, freqs


def extract_mel_spectrogram_stats(y: np.ndarray) -> np.ndarray:
    """对 log-mel 频谱做时间维上的统计（均值+方差），得到一个 1D 向量。"""
    logmel = wav_to_logmel(y)  # [n_mels, time]
    mean = logmel.mean(axis=1)
    std = logmel.std(axis=1)
    return np.concatenate([mean, std], axis=0)  # [2 * n_mels]


def _dct_matrix(n_mfcc: int, n_mels: int) -> np.ndarray:
    """生成用于 MFCC 的 DCT-II 矩阵。"""
    n = np.arange(n_mels)
    k = np.arange(n_mfcc)[:, None]
    dct = np.cos(np.pi * (2 * n + 1) * k / (2.0 * n_mels))
    dct[0] = dct[0] / np.sqrt(2.0)
    dct *= np.sqrt(2.0 / n_mels)
    return dct.astype(np.float32)


def extract_mfcc_with_delta_stats(y: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
    """从 log-mel 提取 MFCC + 一阶/二阶差分，并做时间维统计。"""
    logmel = wav_to_logmel(y)  # [n_mels, time]
    n_mels, t = logmel.shape
    dct = _dct_matrix(n_mfcc, n_mels)  # [n_mfcc, n_mels]
    mfcc = np.dot(dct, logmel)  # [n_mfcc, time]

    def _delta(feat: np.ndarray) -> np.ndarray:
        # 简单相邻差分，保持长度一致（两端复制）
        d = np.zeros_like(feat)
        d[:, 1:-1] = (feat[:, 2:] - feat[:, :-2]) / 2.0
        d[:, 0] = d[:, 1]
        d[:, -1] = d[:, -2]
        return d

    delta = _delta(mfcc)
    delta2 = _delta(delta)

    def _stats(f: np.ndarray) -> np.ndarray:
        return np.concatenate([f.mean(axis=1), f.std(axis=1)], axis=0)  # [2 * n_mfcc]

    mfcc_stats = _stats(mfcc)
    delta_stats = _stats(delta)
    delta2_stats = _stats(delta2)

    return np.concatenate([mfcc_stats, delta_stats, delta2_stats], axis=0)


def extract_spectral_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """频谱质心、带宽等统计特征（对时间求均值+方差）。"""
    mag, freqs = _compute_spectrogram(y, sr)
    power = mag ** 2 + 1e-12
    power_sum = power.sum(axis=0, keepdims=True) + 1e-12  # [1, time]

    # 频谱质心
    centroid = (freqs[:, None] * power).sum(axis=0) / power_sum[0]  # [time]

    # 频谱带宽
    diff = freqs[:, None] - centroid[None, :]
    bandwidth = np.sqrt(((diff ** 2) * power).sum(axis=0) / power_sum[0])  # [time]

    # 频谱滚降点（能量累计到 85% 的频率）
    cumulative = np.cumsum(power, axis=0)
    total = cumulative[-1, :] + 1e-12
    rolloff_freq = np.zeros_like(centroid)
    for t in range(mag.shape[1]):
        idx = np.searchsorted(cumulative[:, t], 0.85 * total[t])
        if idx >= len(freqs):
            idx = len(freqs) - 1
        rolloff_freq[t] = freqs[idx]

    def _stats(f: np.ndarray) -> np.ndarray:
        return np.array([f.mean(), f.std()], dtype=np.float32)

    return np.concatenate(
        [
            _stats(centroid),
            _stats(bandwidth),
            _stats(rolloff_freq),
        ],
        axis=0,
    )


def extract_temporal_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """过零率、能量等时域统计特征。"""
    frame_len = N_FFT
    hop = HOP_LENGTH
    n_frames = 1 + (len(y) - frame_len) // hop if len(y) >= frame_len else 1
    if n_frames <= 0:
        n_frames = 1

    zcr_list = []
    rms_list = []
    for i in range(n_frames):
        start = i * hop
        end = start + frame_len
        if end > len(y):
            frame = np.pad(y[start:], (0, end - len(y)))
        else:
            frame = y[start:end]
        # 过零率
        signs = np.sign(frame)
        signs[signs == 0] = 1
        zc = np.mean(signs[:-1] != signs[1:])
        zcr_list.append(zc)
        # 能量（RMS）
        rms = np.sqrt(np.mean(frame ** 2) + 1e-12)
        rms_list.append(rms)

    zcr_arr = np.array(zcr_list, dtype=np.float32)
    rms_arr = np.array(rms_list, dtype=np.float32)

    def _stats(f: np.ndarray) -> np.ndarray:
        return np.array([f.mean(), f.std()], dtype=np.float32)

    return np.concatenate([_stats(zcr_arr), _stats(rms_arr)], axis=0)


def extract_industrial_specific_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """一些简单的、与设备相关的特征（能量在不同频带的分布等）。"""
    mag, freqs = _compute_spectrogram(y, sr)
    power = mag ** 2
    total_energy = power.sum() + 1e-12

    # 简单划分频带：低频[0,1k]，中频[1k,4k]，高频[4k, sr/2]
    low_mask = freqs < 1000
    mid_mask = (freqs >= 1000) & (freqs < 4000)
    high_mask = freqs >= 4000

    low_energy = power[low_mask].sum() / total_energy
    mid_energy = power[mid_mask].sum() / total_energy
    high_energy = power[high_mask].sum() / total_energy

    # 简单的频谱平坦度（整体）
    mean_spec = power.mean(axis=1) + 1e-12
    geometric_mean = np.exp(np.mean(np.log(mean_spec)))
    arithmetic_mean = np.mean(mean_spec)
    flatness = geometric_mean / (arithmetic_mean + 1e-12)

    return np.array(
        [low_energy, mid_energy, high_energy, flatness],
        dtype=np.float32,
    )


def extract_feature_vector(path: str) -> np.ndarray:
    """综合特征向量：用于工业音频异常检测的手工特征。"""
    y = load_audio(path)

    mel_stats = extract_mel_spectrogram_stats(y)
    mfcc_stats = extract_mfcc_with_delta_stats(y)
    spectral_stats = extract_spectral_features(y)
    temporal_stats = extract_temporal_features(y)
    industrial_stats = extract_industrial_specific_features(y)

    return np.concatenate(
        [mel_stats, mfcc_stats, spectral_stats, temporal_stats, industrial_stats],
        axis=0,
    ).astype(np.float32)


# -----------------------------
# Dataset 定义
# -----------------------------

SECTION_REGEX = re.compile(r"section_(\d\d)")


def parse_section_from_path(path: str) -> int:
    m = SECTION_REGEX.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot parse section from filename: {path}")
    return int(m.group(1))


def parse_is_normal_from_path(path: str) -> int:
    name = os.path.basename(path)
    if "normal" in name:
        return 1
    if "anomaly" in name:
        return 0
    # 对于 train，只包含 normal
    return 1


def debug_print_labels(data_root: str, max_samples: int = 5) -> None:
    """
    简单打印几条训练 / 测试样本及其解析出的标签，方便你检查：
    - section_id 是否和文件名一致
    - normal / anomaly 标签是否正确
    """
    print("==== Debug: train files & section_id ====")
    train_files = collect_train_files(data_root)
    for i, p in enumerate(train_files[:max_samples]):
        sec = parse_section_from_path(p)
        print(f"[train] #{i}: section_id={sec}, path={p}")

    print("\n==== Debug: test files & section_id + is_normal ====")
    pattern_source = os.path.join(data_root, "source_test", "section_*_source_test_*.wav")
    pattern_target = os.path.join(data_root, "target_test", "section_*_target_test_*.wav")
    test_files = sorted(glob(pattern_source) + glob(pattern_target))
    for i, p in enumerate(test_files[:max_samples]):
        sec = parse_section_from_path(p)
        is_normal = parse_is_normal_from_path(p)
        print(f"[test] #{i}: section_id={sec}, is_normal={is_normal}, path={p}")

    if not train_files:
        print("WARNING: no train files found, please check data_root/train.")
    if not test_files:
        print("WARNING: no test files found under source_test / target_test.")


class ValveSectionDataset(Dataset):
    """用于 section 分类的 Dataset（多分类，label=section_id）"""

    def __init__(self, file_list: List[str]):
        self.file_list = file_list

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int):
        path = self.file_list[idx]
        x = extract_feature(path)  # [1, n_mels, time]
        section_id = parse_section_from_path(path)
        return torch.from_numpy(x), torch.tensor(section_id, dtype=torch.long)


class SectionFeatureDataset(Dataset):
    """用于 section 多分类的手工特征 Dataset（label=section_id）。"""

    def __init__(self, file_list: List[str]):
        self.file_list = file_list

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int):
        path = self.file_list[idx]
        fv = extract_feature_vector(path)  # [D]
        section_id = parse_section_from_path(path)
        return torch.from_numpy(fv), torch.tensor(section_id, dtype=torch.long)


class SectionAnomalyDataset(Dataset):
    """用于每个 section 的正常/异常二分类（输入为手工特征向量）。"""

    def __init__(self, file_list: List[str], labels: List[int]):
        assert len(file_list) == len(labels)
        self.file_list = file_list
        self.labels = labels

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int):
        path = self.file_list[idx]
        fv = extract_feature_vector(path)  # [D]
        y = self.labels[idx]
        return torch.from_numpy(fv), torch.tensor(float(y), dtype=torch.float32)


# -----------------------------
# 模型定义
# -----------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        return self.block(x)


class SectionClassifier(nn.Module):
    """简单的 2D CNN，多分类 section_00 / 01 / 02，同时输出特征用于异常检测"""

    def __init__(self, n_sections: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, n_sections)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取用于分类/异常检测的高维特征向量。"""
        x = self.features(x)
        x = self.gap(x)  # [B, C, 1, 1]
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.fc(x)
        return x


class SectionAnomalyMLP(nn.Module):
    """针对单个 section 的正常/异常二分类 MLP，输入为手工特征向量。"""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int] | None = None,
        dropout: float = 0.2,
        use_bn: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        layers: List[nn.Module] = []
        last_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            last_dim = h
        layers.append(nn.Linear(last_dim, 1))  # 输出一个 logit，后续用 sigmoid 转为概率
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D] -> [B]
        return self.net(x).squeeze(-1)


class SectionMLPClassifier(nn.Module):
    """基于手工特征的 section 多分类 MLP。"""

    def __init__(
        self,
        in_dim: int,
        n_sections: int,
        hidden_dims: List[int] | None = None,
        dropout: float = 0.2,
        use_bn: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        layers: List[nn.Module] = []
        last_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            last_dim = h
        layers.append(nn.Linear(last_dim, n_sections))  # 输出每个 section 的 logit
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D] -> [B, n_sections]
        return self.net(x)


class MachineTypeDataset(Dataset):
    """多设备数据集：label 表示属于哪个 data 目录（fan / pump / valve 等）。"""

    def __init__(self, file_list: List[str], labels: List[int]):
        assert len(file_list) == len(labels)
        self.file_list = file_list
        self.labels = labels

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int):
        path = self.file_list[idx]
        x = extract_feature(path)  # [1, n_mels, time]
        y = self.labels[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


class MachineTypeClassifier(nn.Module):
    """设备类型分类 CNN：输入 log-mel 频谱，输出属于哪种设备（fan/pump/valve...）。"""

    def __init__(self, n_machines: int):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, n_machines)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# -----------------------------
# 训练 section 分类器
# -----------------------------

def collect_train_files(data_root: str) -> List[str]:
    pattern = os.path.join(data_root, "train", "section_*_train_normal_*.wav")
    files = glob(pattern)
    if not files:
        raise RuntimeError(f"No train files found with pattern: {pattern}")
    return sorted(files)


def collect_eval_files(data_root: str) -> List[str]:
    """
    用于统一评估 section 分类器的测试集：
    - source_test + target_test 下的所有 section_*_test_*.wav
    这样你可以直接对比不同特征/模型在真实测试域上的最终正确率。
    """
    pattern_source = os.path.join(
        data_root,
        "source_test",
        "section_*_source_test_*.wav",
    )
    pattern_target = os.path.join(
        data_root,
        "target_test",
        "section_*_target_test_*.wav",
    )
    files = glob(pattern_source) + glob(pattern_target)
    return sorted(files)


def simple_train_val_split(
    files: List[str], test_size: float = 0.1, random_state: int = 42
) -> Tuple[List[str], List[str]]:
    """简单实现一个 train/val 划分，避免依赖 scikit-learn。"""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1)")
    n = len(files)
    indices = np.arange(n)
    rng = np.random.RandomState(random_state)
    rng.shuffle(indices)
    n_val = max(1, int(n * test_size))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    train_files = [files[i] for i in train_idx]
    val_files = [files[i] for i in val_idx]
    return train_files, val_files


def train_section_classifier(
    data_root: str,
    batch_size: int = 32,
    num_epochs: int = 10,
    lr: float = 1e-3,
    device: torch.device | None = None,
    model_dir: str | None = None,
    save_path: str | None = None,
):
    # 为每个设备单独建立模型目录，避免 fan/pump/valve 等互相覆盖
    machine_name = infer_machine_name_from_data_root(data_root)
    if model_dir is None:
        model_dir = os.path.join("models", machine_name)
    os.makedirs(model_dir, exist_ok=True)
    if save_path is None:
        save_path = os.path.join(model_dir, "section_classifier.pth")
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    device = device or get_device()

    files = collect_train_files(data_root)
    train_files, val_files = simple_train_val_split(files, test_size=0.1, random_state=42)

    train_ds = ValveSectionDataset(train_files)
    val_ds = ValveSectionDataset(val_files)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # 解析一遍看有多少个不同的 section
    section_ids = sorted({parse_section_from_path(p) for p in files})
    n_sections = max(section_ids) + 1

    model = SectionClassifier(n_sections=n_sections).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        for x, y in tqdm(train_loader, desc=f"[Section] Epoch {epoch}/{num_epochs}", ncols=80):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total += x.size(0)

        train_loss = total_loss / total
        train_acc = total_correct / total

        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += x.size(0)
        val_acc = val_correct / val_total

        print(
            f"[Section] Epoch {epoch}: train_loss={train_loss:.4f}, "
            f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    if best_state is not None:
        torch.save({"state_dict": best_state, "n_sections": n_sections}, save_path)
        print(f"[Section] Saved best model to {save_path}, val_acc={best_val_acc:.4f}")
    else:
        torch.save({"state_dict": model.state_dict(), "n_sections": n_sections}, save_path)
        print(f"[Section] Saved last model to {save_path}")


def evaluate_section_classifier(
    data_root: str,
    batch_size: int = 64,
    device: torch.device | None = None,
    model_path: str = "models/section_classifier.pth",
):
    """
    在 source_test + target_test 上评估 CNN 版 section 分类器的最终正确率。
    方便你和“二进制版 / 手工特征版”直接对比。
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = device or get_device()
    files = collect_eval_files(data_root)
    if not files:
        print(
            "[Section][Eval] No eval files found under "
            f"{os.path.join(data_root, 'source_test')} / "
            f"{os.path.join(data_root, 'target_test')}"
        )
        return 0.0

    ds = ValveSectionDataset(files)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    ckpt = torch.load(model_path, map_location=device)
    n_sections = int(ckpt.get("n_sections", 3))
    model = SectionClassifier(n_sections=n_sections).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    total = 0
    total_correct = 0
    with torch.no_grad():
        for x, y in tqdm(
            loader,
            desc="[Section][Eval] Evaluating CNN section classifier",
            ncols=80,
        ):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total += x.size(0)

    acc = total_correct / max(total, 1)
    print(f"[Section][Eval] accuracy on source_test + target_test: {acc:.4f}")
    return acc


def train_section_mlp_classifier(
    data_root: str,
    batch_size: int = 64,
    num_epochs: int = 20,
    lr: float = 1e-3,
    device: torch.device | None = None,
    model_dir: str | None = None,
    save_path: str | None = None,
):
    """
    使用手工特征（mel/MFCC/频谱/时域/工业特征）训练一个 section 多分类 MLP。
    """
    # 为每个设备单独建立模型目录
    machine_name = infer_machine_name_from_data_root(data_root)
    if model_dir is None:
        model_dir = os.path.join("models", machine_name)
    os.makedirs(model_dir, exist_ok=True)
    if save_path is None:
        save_path = os.path.join(model_dir, "section_mlp_classifier.pth")
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    device = device or get_device()

    files = collect_train_files(data_root)
    train_files, val_files = simple_train_val_split(files, test_size=0.1, random_state=42)

    # 解析 section 数量
    section_ids = sorted({parse_section_from_path(p) for p in files})
    n_sections = max(section_ids) + 1

    # 确定特征维度
    sample_vec = extract_feature_vector(train_files[0])
    in_dim = int(sample_vec.shape[0])

    train_ds = SectionFeatureDataset(train_files)
    val_ds = SectionFeatureDataset(val_files)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SectionMLPClassifier(in_dim=in_dim, n_sections=n_sections).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        for x, y in tqdm(
            train_loader,
            desc=f"[SectionMLP] Epoch {epoch}/{num_epochs}",
            ncols=80,
        ):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total += x.size(0)

        train_loss = total_loss / max(total, 1)
        train_acc = total_correct / max(total, 1)

        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += x.size(0)
        val_acc = val_correct / max(val_total, 1)

        print(
            f"[SectionMLP] Epoch {epoch}: train_loss={train_loss:.4f}, "
            f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    if best_state is None:
        best_state = model.state_dict()

    torch.save(
        {"state_dict": best_state, "n_sections": n_sections, "input_dim": in_dim},
        save_path,
    )
    print(
        f"[SectionMLP] Saved best model to {save_path}, val_acc={best_val_acc:.4f}, in_dim={in_dim}"
    )


def _collect_all_machine_train_files(
    multi_data_root: str,
) -> Tuple[List[str], List[int], List[str]]:
    """
    遍历 multi_data_root 下的所有设备子目录（fan/pump/valve 等），
    收集各自的 train-normal 音频文件，并返回：
    - all_files: 所有训练文件路径
    - all_labels: 与 all_files 对应的设备索引（0..n_machines-1）
    - machine_names: 设备名称列表（索引 -> 名称）
    """
    all_files: List[str] = []
    all_labels: List[int] = []
    machine_names: List[str] = []

    if not os.path.isdir(multi_data_root):
        raise RuntimeError(f"multi_data_root not found: {multi_data_root}")

    for entry in sorted(os.listdir(multi_data_root)):
        machine_dir = os.path.join(multi_data_root, entry)
        if not os.path.isdir(machine_dir):
            continue
        train_dir = os.path.join(machine_dir, "train")
        if not os.path.isdir(train_dir):
            continue

        try:
            machine_files = collect_train_files(machine_dir)
        except RuntimeError:
            # 某些子目录可能不是本任务的数据，直接跳过
            continue
        if not machine_files:
            continue

        mid = len(machine_names)
        machine_names.append(entry)
        all_files.extend(machine_files)
        all_labels.extend([mid] * len(machine_files))

    if not all_files:
        raise RuntimeError(f"No train files found under multi_data_root={multi_data_root}")

    return all_files, all_labels, machine_names


def train_machine_type_classifier(
    multi_data_root: str = "data",
    batch_size: int = 32,
    num_epochs: int = 10,
    lr: float = 1e-3,
    device: torch.device | None = None,
    save_path: str = "models/machine_type_classifier.pth",
):
    """
    训练一个“设备类型分类”模型：
    - 输入任意一段音频（来自 fan / pump / valve / gearbox / ...）
    - 输出属于哪个 data 目录（机器类型）
    """
    device = device or get_device()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    all_files, all_labels, machine_names = _collect_all_machine_train_files(multi_data_root)

    n = len(all_files)
    indices = np.arange(n)
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    n_val = max(1, int(0.1 * n))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_files = [all_files[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    val_files = [all_files[i] for i in val_idx]
    val_labels = [all_labels[i] for i in val_idx]

    train_ds = MachineTypeDataset(train_files, train_labels)
    val_ds = MachineTypeDataset(val_files, val_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    n_machines = len(machine_names)
    print(f"[MachineType] training with {n_machines} device types: {machine_names}")

    model = MachineTypeClassifier(n_machines=n_machines).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        for x, y in tqdm(
            train_loader,
            desc=f"[MachineType] Epoch {epoch}/{num_epochs}",
            ncols=80,
        ):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total += x.size(0)

        train_loss = total_loss / max(total, 1)
        train_acc = total_correct / max(total, 1)

        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += x.size(0)
        val_acc = val_correct / max(val_total, 1)

        print(
            f"[MachineType] Epoch {epoch}: train_loss={train_loss:.4f}, "
            f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    if best_state is None:
        best_state = model.state_dict()

    torch.save(
        {"state_dict": best_state, "machine_names": machine_names},
        save_path,
    )
    print(
        f"[MachineType] Saved best model to {save_path}, "
        f"val_acc={best_val_acc:.4f}, "
        f"machines={machine_names}"
    )


# -----------------------------
# 基于特征中心的异常检测建模（替代自编码器）
# -----------------------------


def collect_section_normal_train_files(data_root: str, section_id: int) -> List[str]:
    """获取某个 section 的所有正常训练样本（source_train + target_train）。"""
    pattern1 = os.path.join(
        data_root,
        "train",
        f"section_{section_id:02d}_source_train_normal_*.wav",
    )
    pattern2 = os.path.join(
        data_root,
        "train",
        f"section_{section_id:02d}_target_train_normal_*.wav",
    )
    files = glob(pattern1) + glob(pattern2)
    return sorted(files)


def collect_section_all_labeled_files(data_root: str, section_id: int) -> Tuple[List[str], List[int]]:
    """收集某个 section 的所有样本（train + source_test + target_test），并打上正常/异常标签。"""
    files: List[str] = []
    labels: List[int] = []

    # 1) 训练集：全是 normal
    train_normal_files = collect_section_normal_train_files(data_root, section_id)
    for p in train_normal_files:
        files.append(p)
        labels.append(1)

    # 2) source_test / target_test：包含 normal 和 anomaly
    pattern_source = os.path.join(
        data_root,
        "source_test",
        f"section_{section_id:02d}_source_test_*.wav",
    )
    pattern_target = os.path.join(
        data_root,
        "target_test",
        f"section_{section_id:02d}_target_test_*.wav",
    )
    test_files = sorted(glob(pattern_source) + glob(pattern_target))
    for p in test_files:
        files.append(p)
        labels.append(parse_is_normal_from_path(p))

    return files, labels


def train_anomaly_mlp_for_section(
    data_root: str,
    section_id: int,
    device: torch.device,
    run_dir: str,
    batch_size: int = 64,
    num_epochs: int = 20,
    lr: float = 1e-3,
) -> Dict[str, object]:
    """为单个 section 训练一个基于多特征的正常/异常 MLP 模型。"""
    files, labels = collect_section_all_labeled_files(data_root, section_id)
    if not files:
        raise RuntimeError(f"No files found for section {section_id:02d}")

    # 打印一下该 section 的正常 / 异常样本统计，方便你判断是否极度不平衡
    labels_arr = np.array(labels, dtype=np.int32)
    num_pos = int(labels_arr.sum())  # label=1 视为 normal
    num_total = int(labels_arr.shape[0])
    num_neg = num_total - num_pos     # label=0 视为 anomaly
    print(
        f"[AnomMLP sec{section_id:02d}] label stats: "
        f"total={num_total}, normal(1)={num_pos}, anomaly(0)={num_neg}"
    )

    # 简单划分 train/val
    n = len(files)
    indices = np.arange(n)
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    n_val = max(1, int(0.2 * n))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_files = [files[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_files = [files[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    # 确定特征维度
    sample_vec = extract_feature_vector(train_files[0])
    in_dim = int(sample_vec.shape[0])

    train_ds = SectionAnomalyDataset(train_files, train_labels)
    val_ds = SectionAnomalyDataset(val_files, val_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SectionAnomalyMLP(in_dim=in_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for x, y in tqdm(
            train_loader,
            desc=f"[AnomMLP sec{section_id:02d}] Epoch {epoch}/{num_epochs}",
            ncols=80,
        ):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total += x.size(0)

        train_loss = total_loss / max(total, 1)

        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                prob = torch.sigmoid(logits)
                pred = (prob >= 0.5).float()
                val_correct += (pred == y).sum().item()
                val_total += x.size(0)
        val_acc = val_correct / max(val_total, 1)

        print(
            f"[AnomMLP sec{section_id:02d}] Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    if best_state is None:
        best_state = model.state_dict()

    # 使用在验证集上的表现自动搜索一个更好的阈值，而不是死用 0.5
    # 目标：在当前 section 的 val 集合上，让 normal/abnormal 的准确率尽量最高
    model.load_state_dict(best_state)
    model.eval()

    all_probs: list[float] = []
    all_labels: list[float] = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            prob = torch.sigmoid(logits)  # p(normal)
            all_probs.extend(prob.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())

    probs_np = np.asarray(all_probs, dtype=np.float32)
    labels_np = np.asarray(all_labels, dtype=np.float32)  # 1=normal, 0=anomaly

    # 如果验证样本太少/异常分布特殊，用一个保守的默认阈值
    best_th = 0.5
    best_th_acc = 0.0
    if probs_np.size > 0:
        # 在 [0.1, 0.9] 之间扫描一批候选阈值，选择准确率最高的
        for th in np.linspace(0.1, 0.9, num=81):
            pred_normal = probs_np >= th
            acc = (pred_normal == (labels_np >= 0.5)).mean()
            if acc > best_th_acc:
                best_th_acc = float(acc)
                best_th = float(th)

    print(
        f"[AnomMLP sec{section_id:02d}] best threshold on val: {best_th:.3f}, "
        f"val_acc@best_th={best_th_acc:.4f}"
    )

    # 保存模型
    model_path = os.path.join(run_dir, f"section_{section_id:02d}_anomaly_mlp.pth")
    torch.save({"state_dict": best_state, "input_dim": in_dim}, model_path)
    print(
        f"[AnomMLP sec{section_id:02d}] Saved best model to {model_path}, "
        f"val_acc={best_val_acc:.4f}"
    )

    return {
        "input_dim": in_dim,
        "threshold": float(best_th),
        "val_acc": float(best_val_acc),
        "num_train": len(train_files),
        "num_val": len(val_files),
    }


def fit_anomaly_models(
    data_root: str,
    model_dir: str | None = None,
    device: torch.device | None = None,
    percentile: float = 95.0,
) -> Dict[str, Dict[str, object]]:
    """为每个 section 训练基于多特征的正常/异常 MLP 模型，并将每次训练结果放在单独子目录。

    注意：这里会根据 data_root 推断设备名称，将结果存放到 models/<machine>/anomaly_runs 下，
    每种设备互不干扰。
    """
    device = device or get_device()

    machine_name = infer_machine_name_from_data_root(data_root)
    if model_dir is None:
        model_dir = os.path.join("models", machine_name)
    os.makedirs(model_dir, exist_ok=True)

    # 每一次训练结果放在一个新的子目录中，便于版本管理
    run_root = os.path.join(model_dir, "anomaly_runs")
    os.makedirs(run_root, exist_ok=True)
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(run_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # 自动推断有哪些 section
    files = collect_train_files(data_root)
    sections = sorted({parse_section_from_path(p) for p in files})

    stats: Dict[str, Dict[str, object]] = {}
    for sec in sections:
        sec_stats = train_anomaly_mlp_for_section(
            data_root=data_root,
            section_id=sec,
            device=device,
            run_dir=run_dir,
        )
        stats[str(sec)] = sec_stats

    out_path = os.path.join(run_dir, "section_anomaly_stats.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 记录本次 run 名称，供推理/评估默认使用“最新一次”
    latest_path = os.path.join(run_root, "latest.txt")
    with open(latest_path, "w", encoding="utf-8") as f:
        f.write(run_name)

    print(f"Saved anomaly stats to {out_path} (run={run_name})")
    return stats


# -----------------------------
# 推理接口：输入一段音频 → section + 正常/异常
# -----------------------------

class ValveAnomalyPipeline:
    def __init__(
        self,
        data_root: str,
        model_dir: str | None = None,
        device: torch.device | None = None,
    ):
        self.data_root = data_root
        # 如果未显式指定模型目录，则根据 data_root 推断设备名称：
        #   data/valve -> models/valve
        #   data/fan   -> models/fan
        if model_dir is None:
            machine_name = infer_machine_name_from_data_root(data_root)
            model_dir = os.path.join("models", machine_name)
        self.model_dir = model_dir
        self.device = device or get_device()

        # 加载 section CNN 分类器
        sec_ckpt = torch.load(
            os.path.join(self.model_dir, "section_classifier.pth"),
            map_location=self.device,
        )
        self.n_sections: int = int(sec_ckpt.get("n_sections", 3))
        self.section_model = SectionClassifier(n_sections=self.n_sections).to(self.device)
        self.section_model.load_state_dict(sec_ckpt["state_dict"])
        self.section_model.eval()

        # 尝试加载基于手工特征的 section MLP 分类器（可选）
        self.section_mlp: SectionMLPClassifier | None = None
        mlp_path = os.path.join(self.model_dir, "section_mlp_classifier.pth")
        if os.path.exists(mlp_path):
            try:
                mlp_ckpt = torch.load(mlp_path, map_location=self.device)
                input_dim = int(mlp_ckpt["input_dim"])
                n_sections_mlp = int(mlp_ckpt.get("n_sections", self.n_sections))
                self.section_mlp = SectionMLPClassifier(
                    in_dim=input_dim,
                    n_sections=n_sections_mlp,
                ).to(self.device)
                self.section_mlp.load_state_dict(mlp_ckpt["state_dict"])
                self.section_mlp.eval()
            except Exception as e:  # 兼容老模型或结构变更
                print(
                    f"[ValveAnomalyPipeline] Warning: failed to load section_mlp from "
                    f"{mlp_path} ({e}). Continue without section_mlp fusion."
                )
                self.section_mlp = None

        # 加载每个 section 的特征中心与阈值（来自最近一次异常模型训练目录）
        run_root = os.path.join(self.model_dir, "anomaly_runs")
        latest_path = os.path.join(run_root, "latest.txt")
        if not os.path.exists(latest_path):
            raise RuntimeError(
                f"未找到 {latest_path}，请先运行 --train_ae（现在用于拟合异常检测模型，每次训练会生成一个新的 run 目录）。"
            )
        with open(latest_path, "r", encoding="utf-8") as f:
            run_name = f.read().strip()
        stats_path = os.path.join(run_root, run_name, "section_anomaly_stats.json")
        if not os.path.exists(stats_path):
            raise RuntimeError(f"未找到 {stats_path}，请确认已成功完成异常模型训练。")
        with open(stats_path, "r", encoding="utf-8") as f:
            raw_stats: Dict[str, Dict[str, object]] = json.load(f)

        # 加载每个 section 的异常检测 MLP 模型和阈值
        self.anomaly_models: Dict[int, SectionAnomalyMLP] = {}
        self.thresholds: Dict[int, float] = {}
        for sec_str, info in raw_stats.items():
            sec = int(sec_str)
            input_dim = int(info["input_dim"])
            th = float(info.get("threshold", 0.5))
            model_path = os.path.join(run_root, run_name, f"section_{sec:02d}_anomaly_mlp.pth")
            if not os.path.exists(model_path):
                raise RuntimeError(f"Missing anomaly model file: {model_path}")
            ckpt = torch.load(model_path, map_location=self.device)
            mlp = SectionAnomalyMLP(in_dim=input_dim).to(self.device)
            mlp.load_state_dict(ckpt["state_dict"])
            mlp.eval()
            self.anomaly_models[sec] = mlp
        self.thresholds[sec] = th

        if not self.anomaly_models:
            print(
                "[ValveAnomalyPipeline] Warning: no anomaly models loaded. "
                "Please ensure you have run --train_ae for this device."
            )

        # 可选：加载特征重要性的“正常基线”（mean/std），用于 z-score 判断
        self.feature_baseline: Dict[int, Dict[str, Dict[str, float]]] = {}
        baseline_path = os.path.join(self.model_dir, "feature_baseline.json")
        if os.path.exists(baseline_path):
            try:
                with open(baseline_path, "r", encoding="utf-8") as f:
                    raw_baseline: Dict[str, Dict[str, Dict[str, float]]] = json.load(f)
                for sec_str, gstats in raw_baseline.items():
                    self.feature_baseline[int(sec_str)] = gstats
            except Exception as e:
                print(
                    f"[ValveAnomalyPipeline] Warning: failed to load feature_baseline "
                    f"from {baseline_path} ({e}). Continue without baseline."
                )

        # 尝试加载特征基线（用于解释各特征类型是否“偏离正常”）
        self.feature_baseline: Dict[str, Dict[str, Dict[str, float]]] = {}
        baseline_path = os.path.join(self.model_dir, "feature_baseline.json")
        if os.path.exists(baseline_path):
            try:
                with open(baseline_path, "r", encoding="utf-8") as f:
                    self.feature_baseline = json.load(f)
            except Exception as e:
                print(
                    f"[ValveAnomalyPipeline] Warning: failed to load feature_baseline "
                    f"from {baseline_path}: {e}"
                )

    @torch.no_grad()
    def predict_section(self, wav_path: str) -> int:
        """
        预测所属 section。
        注意：某些设备的数据集中，可能并不存在所有 section 编号（例如只存在 1、2，没有 0），
        但 CNN 输出的维度仍然是 [0..n_sections-1]。

        为了避免预测到“没有异常模型/阈值的 section”，这里会只在
        self.anomaly_models.keys() 这些有效 section 上选取概率最大的那个。
        """
        # CNN 分支（log-mel 频谱）
        x = extract_feature(wav_path)
        xt = torch.from_numpy(x).unsqueeze(0).to(self.device)
        logits_cnn = self.section_model(xt)  # [1, n_sections]
        prob_cnn = F.softmax(logits_cnn, dim=1).cpu().numpy()[0]

        # 如果没有 hand-crafted MLP，则只用 CNN 概率
        if self.section_mlp is None:
            prob_fused = prob_cnn
        else:
            # MLP 分支（手工特征）
            fv = extract_feature_vector(wav_path)
            xf = torch.from_numpy(fv).unsqueeze(0).to(self.device)  # [1, D]
            logits_mlp = self.section_mlp(xf)  # [1, n_sections]
            prob_mlp = F.softmax(logits_mlp, dim=1).cpu().numpy()[0]
            # 简单平均融合（你可以根据效果再调整权重）
            prob_fused = 0.5 * prob_cnn + 0.5 * prob_mlp

        # 只在“确实训练过异常检测模型”的 section 集合上取最大概率
        valid_secs = sorted(self.anomaly_models.keys())
        best_sec = None
        best_prob = -1.0
        for sec in valid_secs:
            if sec < 0 or sec >= len(prob_fused):
                continue
            p = float(prob_fused[sec])
            if p > best_prob:
                best_prob = p
                best_sec = sec

        if best_sec is None:
            # 理论上不会走到这里，如果走到，退回到全局 argmax 以避免崩溃
            return int(prob_fused.argmax())

        return int(best_sec)

    @torch.no_grad()
    def anomaly_score(self, wav_path: str, section_id: int) -> float:
        if section_id not in self.anomaly_models:
            raise ValueError(f"No anomaly model found for section {section_id}")
        fv = extract_feature_vector(wav_path)
        x = torch.from_numpy(fv).unsqueeze(0).to(self.device)  # [1, D]
        mlp = self.anomaly_models[section_id]
        logit = mlp(x)  # [1]
        prob_normal = torch.sigmoid(logit).item()
        # 将“异常得分”定义为 1 - P(normal)，值越大越异常
        score = 1.0 - prob_normal
        return float(score)

    @torch.no_grad()
    def predict(self, wav_path: str) -> Dict[str, object]:
        sec = self.predict_section(wav_path)
        score = self.anomaly_score(wav_path, sec)
        # 某些老的/不完整的异常模型可能缺失某个 section 的阈值，这里做一下容错
        if sec not in self.thresholds:
            print(
                f"[ValveAnomalyPipeline] Warning: no threshold found for section {sec}, "
                "use default 0.5."
            )
            th = 0.5
        else:
            th = float(self.thresholds[sec])
        is_normal = score <= th
        return {
            "section": sec,
            "anomaly_score": score,
            "threshold": th,
            "is_normal": is_normal,
        }

    @torch.no_grad()
    def predict_with_true_section(self, wav_path: str, section_id: int) -> Dict[str, object]:
        """
        只关心“是否正常”的场景下，如果你已经知道真实的 section_id（例如从文件名/设备信息获得），
        可以直接绕过 section 分类器，只使用对应 section 的异常检测 MLP。

        这样做的好处：
        - 不再受 section 分类错误的影响
        - 最终 normal/abnormal 的准确率理论上可以接近每个 section 上 AnomMLP 的 val_acc（~0.86–0.89）
        """
        if section_id not in self.anomaly_models:
            raise ValueError(f"No anomaly model found for section {section_id}")
        score = self.anomaly_score(wav_path, section_id)
        if section_id not in self.thresholds:
            print(
                f"[ValveAnomalyPipeline] Warning: no threshold found for section {section_id}, "
                "use default 0.5."
            )
            th = 0.5
        else:
            th = float(self.thresholds[section_id])
        is_normal = score <= th
        return {
            "section": section_id,
            "anomaly_score": score,
            "threshold": th,
            "is_normal": is_normal,
        }

    def explain_anomaly(
        self,
        wav_path: str,
        section_id: int | None = None,
        use_true_section: bool = False,
    ) -> Dict[str, object]:
        """
        对单条音频做“异常检测 + 特征类型解释”：
        - 返回正常/异常结果及分数
        - 同时给出各类特征块（mel / MFCC / 频谱 / 时域 / 工业特征）对异常得分的相对重要性

        说明：
        - 不需要重新训练模型，仅基于当前 SectionAnomalyMLP 的梯度进行简单解释
        - importance 值越大，说明该特征块对“异常”的贡献越大
        - 若存在正常基线（mean/std），会计算每一类特征的 z-score，并给出 is_abnormal 标记
        """
        # 1) 选择使用的 section
        if use_true_section and section_id is not None:
            sec = int(section_id)
        else:
            sec = int(self.predict_section(wav_path))

        if sec not in self.anomaly_models:
            raise ValueError(f"No anomaly model found for section {sec}")

        # 2) 提取特征向量，并开启梯度
        fv = extract_feature_vector(wav_path)  # [D] numpy
        x = torch.from_numpy(fv).unsqueeze(0).to(self.device)  # [1, D]
        x.requires_grad_(True)

        mlp = self.anomaly_models[sec]
        mlp.zero_grad()

        # 3) 前向 & 计算异常得分
        logit = mlp(x)  # [1]
        prob_normal = torch.sigmoid(logit)  # [1]
        score = float(1.0 - prob_normal.item())  # 异常得分
        # 这里也做与 predict 一致的容错：某些 section 可能缺少阈值
        if sec not in self.thresholds:
            print(
                f"[ValveAnomalyPipeline] Warning: no threshold found for section {sec} "
                "when explaining anomaly, use default 0.5."
            )
            th = 0.5
        else:
            th = float(self.thresholds[sec])
        is_normal = score <= th

        # 4) 以“异常概率”作为目标，反向传播得到各维特征的重要性
        # 目标越大，说明越异常，因此对 objective 的正向梯度可视作“推动异常”的方向
        objective = 1.0 - prob_normal  # 标量
        objective.backward()

        grads = x.grad.detach().cpu().numpy()[0]  # [D]
        fv_np = fv  # [D]

        # 使用 |grad * feature| 作为简单的重要性度量
        raw_importance = np.abs(grads * fv_np)  # [D]
        total_imp = float(raw_importance.sum())
        if total_imp <= 0.0:
            imp_norm = np.zeros_like(raw_importance, dtype=np.float32)
        else:
            imp_norm = (raw_importance / total_imp).astype(np.float32)

        # 5) 汇总到“特征类型块”级别的贡献
        D = imp_norm.shape[0]
        mel_dim = 2 * N_MELS  # mel 统计特征长度
        # 其余维度根据实现拆分：mfcc_block + spectral(6) + temporal(4) + industrial(4)
        remaining = D - mel_dim
        spectral_dim = 6
        temporal_dim = 4
        industrial_dim = 4
        mfcc_dim = max(remaining - spectral_dim - temporal_dim - industrial_dim, 0)

        idx = 0
        mel_slice = imp_norm[idx: idx + mel_dim]
        idx += mel_dim
        mfcc_slice = imp_norm[idx: idx + mfcc_dim]
        idx += mfcc_dim
        spectral_slice = imp_norm[idx: idx + spectral_dim]
        idx += spectral_dim
        temporal_slice = imp_norm[idx: idx + temporal_dim]
        idx += temporal_dim
        industrial_slice = imp_norm[idx: idx + industrial_dim]

        def _sum(slice_arr: np.ndarray) -> float:
            return float(slice_arr.sum(dtype=np.float32))

        # 先算出各类特征的 importance
        groups_raw = [
            ("mel_stats", _sum(mel_slice)),
            ("mfcc_delta", _sum(mfcc_slice)),
            ("spectral", _sum(spectral_slice)),
            ("temporal", _sum(temporal_slice)),
            ("industrial", _sum(industrial_slice)),
        ]

        # 若存在“正常基线”，使用 z-score + importance 共同判断是否异常
        baseline_for_sec = self.feature_baseline.get(sec, {})

        feature_groups: list[Dict[str, object]] = []
        for name, imp in groups_raw:
            stats = baseline_for_sec.get(name)
            z_score: float | None = None
            is_abnormal = False

            if stats is not None:
                mu = float(stats.get("mean", 0.0))
                std = float(stats.get("std", 1e-6))
                z_score = (imp - mu) / (std + 1e-6)
                # 规则：该特征在该条音频中贡献较大，且相对正常样本显著偏高 → 判为异常
                # importance 阈值 0.3 + z-score 阈值 2.0
                if imp >= 0.3 and z_score >= 2.0:
                    is_abnormal = True
            else:
                # 没有基线时，仅用 importance 做一个简单判断
                if imp >= 0.4:
                    is_abnormal = True

            feature_groups.append(
                {
                    "name": name,
                    "importance": float(imp),
                    "z_score": float(z_score) if z_score is not None else None,
                    "is_abnormal": is_abnormal,
                }
            )

        # 6) 结合 importance + 正常基线，给出 z_score 和 is_abnormal 标记
        baseline_for_section = self.feature_baseline.get(str(sec), {})
        for g in feature_groups:
            name = g["name"]
            imp = float(g["importance"])
            stats = baseline_for_section.get(name)

            if stats is not None:
                mean = float(stats.get("mean", 0.0))
                std = float(stats.get("std", 1e-6))
                p95 = float(stats.get("p95", 1.0))
                if std <= 1e-6:
                    z = 0.0
                else:
                    z = (imp - mean) / std
                # 规则：importance 本身要足够大，并且明显高于“正常”
                is_abnormal_fg = (imp >= 0.3) and (z >= 2.0 or imp >= p95)
            else:
                # 如果还没有基线，就用简单经验规则：importance>=0.4 视为异常
                z = 0.0
                is_abnormal_fg = imp >= 0.4

            g["z_score"] = float(z)
            g["is_abnormal"] = bool(is_abnormal_fg)

        return {
            "section": sec,
            "anomaly_score": score,
            "threshold": th,
            "is_normal": is_normal,
            "feature_groups": feature_groups,
        }

    def compute_feature_baseline(
        self,
        max_files_per_section: int = 200,
        save_path: str | None = None,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        基于“正常样本”统计各特征类型的 importance 分布，用于后续解释：
        - 对每个 section、每个特征组，计算 mean / std / p95 / count
        - 结果写入 feature_baseline.json，并加载到 self.feature_baseline

        只使用当前设备的数据（self.data_root）。
        """
        baseline: Dict[str, Dict[str, Dict[str, float]]] = {}

        sections = sorted(self.anomaly_models.keys())
        if not sections:
            print("[FeatureBaseline] No anomaly models to build baseline for.")
            return baseline

        for sec in sections:
            files, labels = collect_section_all_labeled_files(self.data_root, sec)
            normal_files = [p for p, y in zip(files, labels) if y == 1]
            if not normal_files:
                print(f"[FeatureBaseline] section {sec:02d}: no normal files, skip.")
                continue

            if max_files_per_section > 0 and len(normal_files) > max_files_per_section:
                rng = np.random.RandomState(0)
                idx = rng.choice(len(normal_files), size=max_files_per_section, replace=False)
                normal_files = [normal_files[i] for i in idx]

            group_values: Dict[str, list[float]] = {
                "mel_stats": [],
                "mfcc_delta": [],
                "spectral": [],
                "temporal": [],
                "industrial": [],
            }

            print(
                f"[FeatureBaseline] section {sec:02d}: using {len(normal_files)} normal files "
                "to build feature importance baseline."
            )

            for p in normal_files:
                res = self.explain_anomaly(
                    p,
                    section_id=sec,
                    use_true_section=True,
                )
                for g in res.get("feature_groups", []):
                    name = g.get("name")
                    imp = float(g.get("importance", 0.0))
                    if name in group_values:
                        group_values[name].append(imp)

            section_stats: Dict[str, Dict[str, float]] = {}
            for name, vals in group_values.items():
                if not vals:
                    continue
                arr = np.asarray(vals, dtype=np.float32)
                mean = float(arr.mean())
                std = float(arr.std() + 1e-6)
                p95 = float(np.percentile(arr, 95))
                section_stats[name] = {
                    "mean": mean,
                    "std": std,
                    "p95": p95,
                    "count": float(len(vals)),
                }

            baseline[str(sec)] = section_stats

        if save_path is None:
            save_path = os.path.join(self.model_dir, "feature_baseline.json")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(baseline, f, ensure_ascii=False, indent=2)

        self.feature_baseline = baseline
        print(f"[FeatureBaseline] saved to {save_path}")
        return baseline


class MachineTypePipeline:
    """
    设备类型分类推理管道：
    - 使用 train_machine_type_classifier 训练好的模型
    - 输入任意一段音频，输出预测的设备名称（fan/pump/valve/...）
    """

    def __init__(
        self,
        model_path: str = "models/machine_type_classifier.pth",
        device: torch.device | None = None,
    ):
        self.device = device or get_device()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Machine type model file not found: {model_path}")

        ckpt = torch.load(model_path, map_location=self.device)
        machine_names = ckpt.get("machine_names")
        if machine_names is None:
            raise RuntimeError(
                f"Invalid checkpoint at {model_path}, missing 'machine_names' field."
            )
        self.machine_names: List[str] = list(machine_names)

        n_machines = len(self.machine_names)
        self.model = MachineTypeClassifier(n_machines=n_machines).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict_machine(self, wav_path: str) -> Dict[str, object]:
        x = extract_feature(wav_path)
        xt = torch.from_numpy(x).unsqueeze(0).to(self.device)
        logits = self.model(xt)  # [1, n_machines]
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        mid = int(probs.argmax())
        return {
            "machine_index": mid,
            "machine_name": self.machine_names[mid],
            "probs": probs.tolist(),
        }


def evaluate_machine_type_classifier(
    multi_data_root: str = "data",
    batch_size: int = 64,
    device: torch.device | None = None,
    model_path: str = "models/machine_type_classifier.pth",
) -> float:
    """
    在整个 multi_data_root 下评估“设备类型分类”模型的正确率：
    - multi_data_root 下每个子目录（如 fan/pump/valve/...）视为一个设备类型
    - 对该子目录下所有 wav 文件进行预测
    - 以子目录名作为真值标签，统计整体准确率 & 每类准确率
    """
    device = device or get_device()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Machine type model file not found: {model_path}")

    ckpt = torch.load(model_path, map_location=device)
    machine_names = ckpt.get("machine_names")
    if machine_names is None:
        raise RuntimeError(
            f"Invalid checkpoint at {model_path}, missing 'machine_names' field."
        )
    machine_names = list(machine_names)
    n_machines = len(machine_names)
    name_to_idx = {name: i for i, name in enumerate(machine_names)}

    # 收集所有需要评估的 wav 文件及其真实机器标签
    all_files: List[str] = []
    all_labels: List[int] = []
    per_machine_files: Dict[int, int] = {i: 0 for i in range(n_machines)}

    for name in machine_names:
        machine_dir = os.path.join(multi_data_root, name)
        if not os.path.isdir(machine_dir):
            print(f"[MachineType][Eval] skip {name}, dir not found: {machine_dir}")
            continue
        # 递归拿这个机器目录下的所有 wav（train/source_test/target_test 全部）
        pattern = os.path.join(machine_dir, "**", "*.wav")
        files = glob(pattern, recursive=True)
        if not files:
            print(f"[MachineType][Eval] no wav files under {machine_dir}, skip.")
            continue

        mid = name_to_idx[name]
        all_files.extend(files)
        all_labels.extend([mid] * len(files))
        per_machine_files[mid] += len(files)

    if not all_files:
        raise RuntimeError(
            f"[MachineType][Eval] No wav files found under multi_data_root={multi_data_root}"
        )

    print(
        "[MachineType][Eval] evaluating on total "
        f"{len(all_files)} files from devices: {machine_names}"
    )

    ds = MachineTypeDataset(all_files, all_labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = MachineTypeClassifier(n_machines=n_machines).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    total = 0
    total_correct = 0
    per_machine_correct: Dict[int, int] = {i: 0 for i in range(n_machines)}

    with torch.no_grad():
        for x, y in tqdm(
            loader,
            desc="[MachineType][Eval] Evaluating device classifier",
            ncols=80,
        ):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)

            total_correct += (preds == y).sum().item()
            total += x.size(0)

            for yi, pi in zip(y.cpu().tolist(), preds.cpu().tolist()):
                if yi == pi:
                    per_machine_correct[int(yi)] += 1

    overall_acc = total_correct / max(total, 1)
    print(f"[MachineType][Eval] overall accuracy on all machines: {overall_acc:.4f}")

    # 每种设备单独准确率
    print("[MachineType][Eval] per-machine accuracy:")
    for i, name in enumerate(machine_names):
        n_i = per_machine_files.get(i, 0)
        c_i = per_machine_correct.get(i, 0)
        if n_i == 0:
            acc_i = 0.0
        else:
            acc_i = c_i / n_i
        print(f"  - {name}: acc={acc_i:.4f}  ({c_i}/{n_i})")

    return overall_acc


# -----------------------------
# 示例 main：先训练，再推理一条音频
# -----------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Valve industrial audio anomaly detection")
    parser.add_argument("--data_root", type=str, default="data/valve", help="数据集根目录（包含 train/source_test/target_test）")
    parser.add_argument("--device", type=str, default="cuda", help="cuda 或 cpu")
    parser.add_argument("--train_section", action="store_true", help="训练 section 分类模型")
    parser.add_argument(
        "--train_section_mlp",
        action="store_true",
        help="使用多种工业音频手工特征训练 section MLP 分类模型（用于与 CNN 融合）",
    )
    parser.add_argument(
        "--eval_section",
        action="store_true",
        help="在 source_test + target_test 上评估 section CNN 分类器的分类准确率",
    )
    parser.add_argument(
        "--train_ae",
        action="store_true",
        help="为每个 section 训练基于多种工业音频特征的正常/异常 MLP 模型（结果存入新的 run 目录）",
    )
    parser.add_argument("--infer_example", type=str, default="", help="推理示例音频路径（可选）")
    parser.add_argument(
        "--train_machine_classifier",
        action="store_true",
        help="训练一个设备类型分类模型（输入任意音频，判断属于 fan/pump/valve/... 中的哪一类）",
    )
    parser.add_argument(
        "--multi_data_root",
        type=str,
        default="data",
        help="包含多个设备子目录的根目录（默认 'data'，其下有 fan/gearbox/pump/.../valve）",
    )
    parser.add_argument(
        "--infer_machine",
        type=str,
        default="",
        help="使用已训练的设备类型分类模型，预测该音频属于哪个 data 目录",
    )
    parser.add_argument(
        "--eval_machine_classifier",
        action="store_true",
        help="在 multi_data_root 下的所有设备数据上，评估设备类型分类模型的整体和逐类准确率",
    )
    parser.add_argument(
        "--compute_feature_baseline",
        action="store_true",
        help="基于正常样本为当前设备计算特征重要性基线（用于后续解释特征是否异常）",
    )
    parser.add_argument(
        "--debug_labels",
        action="store_true",
        help="打印几条训练/测试音频的解析标签（section_id, is_normal），用于排查标签是否有问题",
    )
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Using device: {device}")

    # 如果只想快速检查标签解析是否正确，可以加上 --debug_labels
    if args.debug_labels:
        debug_print_labels(args.data_root)
        return

    if args.train_section:
        train_section_classifier(
            data_root=args.data_root,
            device=device,
        )

    if args.train_section_mlp:
        train_section_mlp_classifier(
            data_root=args.data_root,
            device=device,
        )

    if args.eval_section:
        # 根据 data_root 推断设备名，并从对应 models/<machine>/section_classifier.pth 加载
        machine_name = infer_machine_name_from_data_root(args.data_root)
        model_path = os.path.join("models", machine_name, "section_classifier.pth")
        evaluate_section_classifier(
            data_root=args.data_root,
            device=device,
            model_path=model_path,
        )

    if args.train_ae:
        fit_anomaly_models(
            data_root=args.data_root,
            device=device,
        )

    if args.infer_example:
        pipeline = ValveAnomalyPipeline(
            data_root=args.data_root,
            device=device,
        )
        result = pipeline.predict(args.infer_example)
        print("Inference result:")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.train_machine_classifier:
        train_machine_type_classifier(
            multi_data_root=args.multi_data_root,
            device=device,
        )

    if args.infer_machine:
        mt_pipeline = MachineTypePipeline(
            device=device,
        )
        mt_result = mt_pipeline.predict_machine(args.infer_machine)
        print("Machine classifier result:")
        print(json.dumps(mt_result, ensure_ascii=False, indent=2))

    if args.eval_machine_classifier:
        evaluate_machine_type_classifier(
            multi_data_root=args.multi_data_root,
            device=device,
        )

    if args.compute_feature_baseline:
        pipeline = ValveAnomalyPipeline(
            data_root=args.data_root,
            device=device,
        )
        pipeline.compute_feature_baseline()


if __name__ == "__main__":
    main()


