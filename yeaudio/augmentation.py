# @Time    : 2024-08-30
# @Author  : yeyupiaoling
import os
import random

import numpy as np
from PIL import Image
from PIL.Image import BICUBIC
from loguru import logger
from yeaudio.audio import AudioSegment

__all__ = ["SpeedPerturbAugmentor", "VolumePerturbAugmentor", "ShiftPerturbAugmentor",
           "ResampleAugmentor", "NoisePerturbAugmentor", "ReverbPerturbAugmentor",
           "SpecAugmentor", "SpecSubAugmentor"]


class SpeedPerturbAugmentor(object):
    """随机语速扰动的音频数据增强器

    :param prob: 数据增强概率
    :type prob: float
    :param speed_perturb_3_class: 是否使用语速三类语速增强，只在声纹识别项目上使用
    :type speed_perturb_3_class: bool
    :param num_speakers: 说话人数量，只在声纹识别项目上使用
    :type num_speakers: int
    """

    def __init__(self, prob=0.0, speed_perturb_3_class=False, num_speakers=None):
        self.speeds = [1.0, 0.9, 1.1]
        self.prob = prob
        self.num_speakers = num_speakers
        self.speed_perturb_3_class = speed_perturb_3_class
        if self.speed_perturb_3_class:
            assert self.num_speakers is not None, "使用语速三类语速增强的话，需要设置num_speakers参数"

    def __call__(self, audio_segment: AudioSegment, spk_id: int = None) -> AudioSegment or [AudioSegment, int]:
        if random.random() < self.prob:
            speed_idx = random.randint(0, 2)
            speed_rate = self.speeds[speed_idx]
            if speed_rate != 1.0:
                audio_segment.change_speed(speed_rate)
            # 注意使用语速增强分类数量会大三倍
            if spk_id is not None and self.speed_perturb_3_class:
                spk_id = spk_id + self.num_speakers * speed_idx
        if spk_id is not None:
            return audio_segment, spk_id
        else:
            return audio_segment


class VolumePerturbAugmentor(object):
    """随机音量扰动的音频数据增强器

    :param prob: 数据增强概率
    :type prob: float
    :param min_gain_dBFS: 最小音量，单位为分贝。
    :type min_gain_dBFS: int
    :param max_gain_dBFS: 最大音量，单位为分贝。
    :type max_gain_dBFS: int
    """

    def __init__(self, prob=0.0, min_gain_dBFS=-15, max_gain_dBFS=15):
        self.prob = prob
        self.min_gain_dBFS = min_gain_dBFS
        self.max_gain_dBFS = max_gain_dBFS

    def __call__(self, audio_segment: AudioSegment) -> AudioSegment:
        if random.random() < self.prob:
            gain = random.uniform(self.min_gain_dBFS, self.max_gain_dBFS)
            audio_segment.gain_db(gain)
        return audio_segment


class ShiftPerturbAugmentor(object):
    """添加随机位移扰动的音频数增强器

    :param prob: 数据增强概率
    :type prob: float
    :param min_shift_ms: 最小偏移，单位为毫秒。
    :type min_shift_ms: int
    :param max_shift_ms: 最大偏移，单位为毫秒。
    :type max_shift_ms: int
    """

    def __init__(self, prob=0.0, min_shift_ms=-5, max_shift_ms=5):
        self.prob = prob
        self._min_shift_ms = min_shift_ms
        self._max_shift_ms = max_shift_ms

    def __call__(self, audio_segment: AudioSegment) -> AudioSegment:
        if random.random() < self.prob:
            shift_ms = random.uniform(self._min_shift_ms, self._max_shift_ms)
            audio_segment.shift(shift_ms)
        return audio_segment


class ResampleAugmentor(object):
    """随机重采样的音频数据增强器

    :param prob: 数据增强概率
    :type prob: float
    :param new_sample_rate: 新采样率列表
    :type new_sample_rate: list
    """

    def __init__(self, prob=0.0, new_sample_rate=(8000, 16000, 24000)):
        self.prob = prob
        self._new_sample_rate = new_sample_rate

    def __call__(self, audio_segment: AudioSegment) -> AudioSegment:
        if random.random() < self.prob:
            _new_sample_rate = np.random.choice(self._new_sample_rate)
            audio_segment.resample(_new_sample_rate)
        return audio_segment


class NoisePerturbAugmentor(object):
    """随机噪声扰动的音频数据增强器

    :param noise_dir: 噪声文件夹路径，该文件夹下是噪声音频文件
    :type noise_dir: str
    :param prob: 数据增强概率
    :type prob: float
    :param min_snr_dB: 最小信噪比
    :type min_snr_dB: int
    :param max_snr_dB: 最大信噪比
    :type max_snr_dB: int
    """

    def __init__(self, noise_dir='', prob=0.0, min_snr_dB=10, max_snr_dB=50):
        self.prob = prob
        self.min_snr_dB = min_snr_dB
        self.max_snr_dB = max_snr_dB
        self.noises_path = self.get_audio_path(path=noise_dir)
        logger.info(f"噪声增强的噪声音频文件数量: {len(self.noises_path)}")

    def __call__(self, audio_segment: AudioSegment) -> AudioSegment:
        if len(self.noises_path) > 0 and random.random() < self.prob:
            # 随机选择一个noises_path中的一个
            noise_file = random.sample(self.noises_path, 1)[0]
            # 随机生成snr_dB的值
            snr_dB = random.uniform(self.min_snr_dB, self.max_snr_dB)
            # 将噪声添加到audio_segment中，snr_dB是噪声的增益
            audio_segment.add_noise(noise_file, snr_dB)
        return audio_segment

    # 获取文件夹下的全部音频文件路径
    @staticmethod
    def get_audio_path(path):
        if path is None or not os.path.exists(path):
            return []
        paths = []
        for file in os.listdir(path):
            paths.append(os.path.join(path, file))
        return paths


class ReverbPerturbAugmentor(object):
    """随机混响的音频数据增强器

    :param reverb_dir: 混响文件夹路径，该文件夹下是噪声音频文件
    :type reverb_dir: str
    :param prob: 数据增强概率
    :type prob: float
    """

    def __init__(self, reverb_dir='', prob=0.0):
        self.prob = prob
        self.reverb_path = self.get_audio_path(path=reverb_dir)
        logger.info(f"混响增强音频文件数量: {len(self.reverb_path)}")

    def __call__(self, audio_segment: AudioSegment) -> AudioSegment:
        if len(self.reverb_path) > 0 and random.random() < self.prob:
            # 随机选择混响音频
            reverb_file = random.sample(self.reverb_path, 1)[0]
            # 生成混响音效
            audio_segment.reverb(reverb_file)
        return audio_segment

    # 获取文件夹下的全部音频文件路径
    @staticmethod
    def get_audio_path(path):
        if path is None or not os.path.exists(path):
            return []
        paths = []
        for file in os.listdir(path):
            paths.append(os.path.join(path, file))
        return paths


class SpecAugmentor(object):
    """频域掩蔽和时域掩蔽的音频特征数据增强器
    论文：https://arxiv.org/abs/1904.08779
    论文：https://arxiv.org/abs/1912.05533


    :param prob: 数据增强概率
    :type prob: float
    :param freq_mask_ratio: 频域掩蔽的比例
    :type freq_mask_ratio: float
    :param n_freq_masks: 频域掩蔽次数
    :type n_freq_masks: int
    :param time_mask_ratio: 时间掩蔽的比例
    :type time_mask_ratio: float
    :param n_time_masks: 时间掩蔽次数
    :type n_time_masks: int
    :param inplace: 用结果覆盖
    :type inplace: bool
    :param replace_with_zero: 是否使用0作为掩码，否则使用平均值
    :type replace_with_zero: bool
    """

    def __init__(self,
                 prob=0.0,
                 freq_mask_ratio=0.15,
                 n_freq_masks=2,
                 time_mask_ratio=0.05,
                 n_time_masks=2,
                 inplace=True,
                 max_time_warp=5,
                 replace_with_zero=False):
        self.prob = prob
        self.max_f_ratio = freq_mask_ratio
        self.n_freq_masks = n_freq_masks
        self.max_t_ratio = time_mask_ratio
        self.n_time_masks = n_time_masks
        self.inplace = inplace
        self.max_time_warp = max_time_warp
        self.replace_with_zero = replace_with_zero

    def time_warp(self, x):
        """对时间维度扭曲"""
        window = self.max_time_warp
        if window == 0:
            return x

        t = x.shape[0]
        if t - window <= window:
            return x
        # NOTE: randrange(a, b) emits a, a + 1, ..., b - 1
        center = random.randrange(window, t - window)
        warped = random.randrange(center - window, center + window) + 1  # 1 ... t - 1
        left = Image.fromarray(x[:center]).resize((x.shape[1], warped), BICUBIC)
        right = Image.fromarray(x[center:]).resize((x.shape[1], t - warped), BICUBIC)
        if self.inplace:
            x[:warped] = left
            x[warped:] = right
            return x
        return np.concatenate((left, right), 0)

    def freq_mask(self, x):
        """频域掩蔽"""
        freq_len = x.shape[1]
        mask_freq_len = int(freq_len * self.max_f_ratio)
        for i in range(self.n_freq_masks):
            start = random.randint(0, freq_len - 1)
            length = random.randint(1, mask_freq_len)
            end = min(freq_len, start + length)
            if self.replace_with_zero:
                x[:, start:end] = 0
            else:
                x[:, start:end] = x.mean()
        return x

    def time_mask(self, x):
        """时域掩蔽"""
        time_len = x.shape[0]
        mask_time_len = int(time_len * self.max_t_ratio)
        for i in range(self.n_time_masks):
            start = random.randint(0, time_len - 1)
            length = random.randint(1, mask_time_len)
            end = min(time_len, start + length)
            if self.replace_with_zero:
                x[start:end, :] = 0
            else:
                x[start:end, :] = x.mean()
        return x

    def __call__(self, x) -> np.ndarray:
        """
        param x: spectrogram (time, freq)
        type x: np.ndarray
        """
        if random.random() < self.prob:
            assert isinstance(x, np.ndarray)
            assert x.ndim == 2
            x = self.time_warp(x)
            x = self.freq_mask(x)
            x = self.time_mask(x)
        return x


class SpecSubAugmentor(object):
    """从原始音频中随机替换部分帧，以模拟语音的时移。
    论文：https://arxiv.org/abs/2106.05642

    :param prob: 数据增强概率
    :type prob: float
    :param max_time: 时间替换的最大宽度
    :type max_time: int
    :param num_time_sub: 时间替换的的次数
    :type num_time_sub: int
    """

    def __init__(self, prob=0.0, max_time=20, num_time_sub=3):
        self.prob = prob
        self.max_time = max_time
        self.num_time_sub = num_time_sub

    def __call__(self, x) -> np.ndarray:
        """
        param x: spectrogram (time, freq)
        type x: np.ndarray
        """
        y = x.copy()
        max_frames = y.shape[0]
        for i in range(self.num_time_sub):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, self.max_time)
            end = min(max_frames, start + length)
            pos = random.randint(0, start)
            y[start:end, :] = x[start - pos:end - pos, :]
        return y
