# @Time    : 2024-08-28
# @Author  : yeyupiaoling
import io
import os
import random
from typing import List, Dict

import numpy as np
import resampy
import soundfile
from scipy import signal

from yeaudio.utils.av_utils import buf_to_float, decode_audio


class AudioSegment(object):
    def __init__(self, samples, sample_rate):
        """创建单通道音频片段实例

        :param samples: 音频数据，维度为[num_samples x num_channels]
        :type samples: ndarray.float32
        :param sample_rate: 音频的采样率
        :type sample_rate: int
        """
        self.vad_model = None
        self._samples = self._convert_samples_to_float32(samples)
        self._sample_rate = sample_rate
        if self._samples.ndim >= 2:
            self._samples = np.mean(self._samples, 1)

    def __eq__(self, other):
        """返回两个实例是否相等

        :param other: 比较的另一个音频片段实例
        :type other: AudioSegment
        """
        if type(other) is not type(self):
            return False
        if self.sample_rate != other.sample_rate:
            return False
        if self.samples.shape != other.samples.shape:
            return False
        if np.any(self.samples != other.samples):
            return False
        return True

    def __ne__(self, other):
        """返回两个实例是否不相等

        :param other: 比较的另一个音频片段实例
        :type other: AudioSegment
        """
        return not self.__eq__(other)

    def __str__(self):
        """返回该音频的信息"""
        return (f"{type(self)}: num_samples={self.num_samples}, sample_rate={self.sample_rate}, "
                f"duration={self.duration:.3f}sec, rms={self.rms_db:.2f}dB")

    @classmethod
    def from_file(cls, file):
        """从音频文件创建音频段，支持wav、mp3、mp4等多种音频格式

        :param file: 文件路径，或者文件对象
        :type file: str, BufferedReader
        :return: 音频片段实例
        :rtype: AudioSegment
        """
        assert os.path.exists(file), f'文件不存在，请检查路径：{file}'
        try:
            samples, sample_rate = soundfile.read(file, dtype='float32')
        except:
            # 支持更多格式数据
            samples, sample_rate = decode_audio(file=file)
        return cls(samples, sample_rate)

    @classmethod
    def slice_from_file(cls, file, start=None, end=None):
        """只加载一小段音频，而不需要将整个文件加载到内存中，这是非常浪费的。

        :param file: 输入音频文件路径或文件对象
        :type file: str|file
        :param start: 开始时间，单位为秒。如果start是负的，则它从末尾开始计算。如果没有提供，这个函数将从最开始读取。
        :type start: float
        :param end: 结束时间，单位为秒。如果end是负的，则它从末尾开始计算。如果没有提供，默认的行为是读取到文件的末尾。
        :type end: float
        :return: AudioSegment输入音频文件的指定片的实例。
        :rtype: AudioSegment
        :raise ValueError: 如果开始或结束的设定不正确，则会抛出ValueError异常
        """
        sndfile = soundfile.SoundFile(file)
        sample_rate = sndfile.samplerate
        duration = round(float(len(sndfile)) / sample_rate, 3)
        start = 0. if start is None else round(start, 3)
        end = duration if end is None else round(end, 3)
        # 从末尾开始计
        if start < 0.0: start += duration
        if end < 0.0: end += duration
        # 保证数据不越界
        if start < 0.0: start = 0.0
        if end > duration: end = duration
        if end < 0.0:
            raise ValueError(f"切片结束位置({end} s)越界")
        if start > end:
            raise ValueError(f"切片开始位置({start} s)晚于切片结束位置({end} s)")
        start_frame = int(start * sample_rate)
        end_frame = int(end * sample_rate)
        sndfile.seek(start_frame)
        data = sndfile.read(frames=end_frame - start_frame, dtype='float32')
        return cls(data, sample_rate)

    @classmethod
    def from_bytes(cls, data):
        """从wav格式的音频字节创建音频段

        :param data: 包含音频样本的字节
        :type data: bytes
        :return: 音频片段实例
        :rtype: AudioSegment
        """
        samples, sample_rate = soundfile.read(io.BytesIO(data), dtype='float32')
        return cls(samples, sample_rate)

    @classmethod
    def from_pcm_bytes(cls, data, channels=1, samp_width=2, sample_rate=16000):
        """从包含无格式PCM音频的字节创建音频

        :param data: 包含音频样本的字节
        :type data: bytes
        :param channels: 音频的通道数
        :type channels: int
        :param samp_width: 音频采样的宽度，如np.int16为2
        :type samp_width: int
        :param sample_rate: 音频样本采样率
        :type sample_rate: int
        :return: 音频片段实例
        :rtype: AudioSegment
        """
        samples = buf_to_float(data, n_bytes=samp_width)
        if channels > 1:
            samples = samples.reshape(-1, channels)
        return cls(samples, sample_rate)

    @classmethod
    def from_ndarray(cls, data, sample_rate=16000):
        """从numpy.ndarray创建音频段

        :param data: numpy.ndarray类型的音频数据
        :type data: ndarray
        :param sample_rate: 音频样本采样率
        :type sample_rate: int
        :return: 音频片段实例
        :rtype: AudioSegment
        """
        return cls(data, sample_rate)

    @classmethod
    def concatenate(cls, *segments):
        """将任意数量的音频片段连接在一起

        :param segments: 输入音频片段被连接
        :type segments: AudioSegment
        :return: 拼接后的音频段实例
        :rtype: AudioSegment
        :raises ValueError: 如果音频实例列表为空或者采样率不一致，则会抛出ValueError异常
        :raises TypeError: 如果输入的片段类型不一致，则会抛出TypeError异常
        """
        # Perform basic sanity-checks.
        if len(segments) == 0:
            raise ValueError("没有音频片段被给予连接")
        sample_rate = segments[0].sample_rate
        for seg in segments:
            if sample_rate != seg.sample_rate:
                raise ValueError("能用不同的采样率连接片段")
            if type(seg) is not cls:
                raise TypeError("只有相同类型的音频片段可以连接")
        samples = np.concatenate([seg.samples for seg in segments])
        return cls(samples, sample_rate)

    @classmethod
    def make_silence(cls, duration, sample_rate):
        """创建给定持续时间和采样率的静音音频段

        :param duration: 静音的时间，以秒为单位
        :type duration: float
        :param sample_rate: 音频采样率
        :type sample_rate: int
        :return: 给定持续时间的静音AudioSegment实例
        :rtype: AudioSegment
        """
        samples = np.zeros(int(duration * sample_rate))
        return cls(samples, sample_rate)

    def to_wav_file(self, filepath, dtype='float32'):
        """保存音频段到磁盘为wav文件

        :param filepath: WAV文件路径或文件对象，以保存音频段
        :type filepath: str|file
        :param dtype: 音频数据类型，可选: 'int16', 'int32', 'float32', 'float64'
        :type dtype: str
        :raises TypeError: 如果类型不支持，则会抛出TypeError异常
        """
        samples = self._convert_samples_from_float32(self._samples, dtype)
        subtype_map = {
            'int16': 'PCM_16',
            'int32': 'PCM_32',
            'float32': 'FLOAT',
            'float64': 'DOUBLE'
        }
        soundfile.write(
            filepath,
            samples,
            self._sample_rate,
            format='WAV',
            subtype=subtype_map[dtype])

    def superimpose(self, other):
        """将另一个段的样本添加到这个段的样本中(以样本方式添加，而不是段连接)。

        :param other: 包含样品的片段被添加进去
        :type other: AudioSegments
        :raise ValueError: 如果两段音频采样率或者长度不一致，则会抛出ValueError异常
        :raise TypeError: 如果两个片段的类型不匹配，则会抛出TypeError异常
        """
        if not isinstance(other, type(self)):
            raise TypeError(f"不能添加不同类型的段: {type(self)} 和 {type(other)}")
        if self.sample_rate != other.sample_rate:
            raise ValueError("采样率必须匹配才能添加片段")
        if len(self.samples) != len(other.samples):
            raise ValueError("段长度必须匹配才能添加段")
        self._samples += other._samples

    def to_bytes(self, dtype='float32'):
        """创建音频内容的字节

        :param dtype: 导出样本的数据类型。可选: 'int16', 'int32', 'float32', 'float64'.
        :type dtype: str
        :return: 音频内容的字节
        :rtype: bytes
        """
        samples = self._convert_samples_from_float32(self._samples, dtype)
        return samples.tostring()

    def to_pcm_bytes(self):
        """创建pcm格式的字节

        :return: pcm格式的字节
        :rtype: bytes
        """
        return self.to_bytes(dtype='int16')

    def to(self, dtype='int16'):
        """类型转换

        :param dtype: 导出样本的数据类型。可选: 'int16', 'int32', 'float32', 'float64'
        :type dtype: str
        :return: np.ndarray containing `dtype` audio content.
        :rtype: np.ndarray
        """
        samples = self._convert_samples_from_float32(self._samples, dtype)
        return samples

    def gain_db(self, gain):
        """对音频施加分贝增益。

        :param gain: 用于样品的分贝增益
        :type gain: float|1darray
        """
        self._samples *= 10. ** (gain / 20.)

    def change_speed(self, speed_rate):
        """通过线性插值改变音频速度

        :param speed_rate: 修改的音频速率:
                           speed_rate > 1.0, 加快音频速度;
                           speed_rate = 1.0, 音频速度不变;
                           speed_rate < 1.0, 减慢音频速度;
                           speed_rate <= 0.0, 错误数值.
        :type speed_rate: float
        :raises ValueError: 如果速度速率小于或等于0，则引发ValueError
        """
        if speed_rate == 1.0:
            return
        if speed_rate <= 0:
            raise ValueError("速度速率应大于零")
        old_length = self._samples.shape[0]
        new_length = int(old_length / speed_rate)
        old_indices = np.arange(old_length)
        new_indices = np.linspace(start=0, stop=old_length, num=new_length)
        self._samples = np.interp(new_indices, old_indices, self._samples).astype(np.float32)

    def normalize(self, target_db=-20, max_gain_db=300.0):
        """将音频归一化，使其具有所需的有效值(以分贝为单位)

        :param target_db: 目标均方根值，单位为分贝。这个值应该小于0.0，因为0.0是全尺寸音频。
        :type target_db: float
        :param max_gain_db: 最大允许的增益值，单位为分贝，这是为了防止在对全0信号进行归一化时出现Nan值。
        :type max_gain_db: float
        :raises ValueError: 如果所需的增益大于max_gain_db，则引发ValueError
        """
        gain = target_db - self.rms_db
        if gain > max_gain_db:
            raise ValueError(f"无法将段规范化到{target_db}dB，音频增益{gain}增益已经超过max_gain_db ({max_gain_db}dB)")
        self.gain_db(min(max_gain_db, target_db - self.rms_db))

    def resample(self, target_sample_rate, filter='kaiser_best'):
        """按目标采样率重新采样音频

        :param target_sample_rate: 重采样的目标采样率
        :type target_sample_rate: int
        :param filter: 使用的重采样滤波器，支持'kaiser_best'、'kaiser_fast'
        :type filter: str
        """
        if self.sample_rate == target_sample_rate: return
        self._samples = resampy.resample(self.samples, self.sample_rate, target_sample_rate, filter=filter)
        self._sample_rate = target_sample_rate

    def pad_silence(self, duration, sides='both'):
        """在这个音频样本上加一段静音

        :param duration: 静默段的持续时间(以秒为单位)
        :type duration: float
        :param sides: 添加的位置:
                     'beginning' - 在开始位置前增加静音段;
                     'end' - 在结束位置增加静音段;
                     'both' - 在开始和结束位置都增加静音段.
        :type sides: str
        :raises ValueError: 如果sides的值不是beginning、end或both，则引发ValueError
        """
        if duration == 0.0:
            return self
        cls = type(self)
        silence = self.make_silence(duration, self._sample_rate)
        if sides == "beginning":
            padded = cls.concatenate(silence, self)
        elif sides == "end":
            padded = cls.concatenate(self, silence)
        elif sides == "both":
            padded = cls.concatenate(silence, self, silence)
        else:
            raise ValueError(f"Unknown value for the sides {sides}")
        self._samples = padded._samples

    def pad(self, pad_width, mode='wrap', **kwargs):
        """在这个音频样本上加一段音频，等同numpy.pad

        param pad_width: Padding width.
        :type pad_width: sequence|array_like|int
        :param mode: 填充模式
        :type mode: str|function|optional
        """
        self._samples = np.pad(self.samples, pad_width=pad_width, mode=mode, **kwargs)

    def shift(self, shift_ms):
        """音频偏移。如果shift_ms为正，则随时间提前移位;如果为负，则随时间延迟移位。填补静音以保持持续时间不变。

        :param shift_ms: 偏移时间。如果是正的，随时间前进；如果负，延时移位。
        :type shift_ms: float
        :raises ValueError: 如果shift_ms的绝对值大于音频持续时间，则引发ValueError
        """
        if abs(shift_ms) / 1000.0 > self.duration:
            raise ValueError("shift_ms的绝对值应该小于音频持续时间")
        shift_samples = int(shift_ms * self._sample_rate / 1000)
        if shift_samples > 0:
            # time advance
            self._samples[:-shift_samples] = self._samples[shift_samples:]
            self._samples[-shift_samples:] = 0
        elif shift_samples < 0:
            # time delay
            self._samples[-shift_samples:] = self._samples[:shift_samples]
            self._samples[:-shift_samples] = 0

    def subsegment(self, start_sec=None, end_sec=None):
        """在给定的边界之间切割音频片段

        :param start_sec: 开始裁剪的位置，以秒为单位，默认为0
        :type start_sec: float
        :param end_sec: 结束裁剪的位置，以秒为单位，默认为音频长度
        :type end_sec: float
        :raise ValueError: 如果start_sec或end_sec的值越界，则引发ValueError
        """
        start_sec = 0.0 if start_sec is None else start_sec
        end_sec = self.duration if end_sec is None else end_sec
        if start_sec < 0.0:
            start_sec = self.duration + start_sec
        if end_sec < 0.0:
            end_sec = self.duration + end_sec
        if start_sec < 0.0:
            raise ValueError(f"切片起始位置({start_sec} s)越界")
        if end_sec < 0.0:
            raise ValueError(f"切片结束位置({end_sec} s)越界")
        if start_sec > end_sec:
            raise ValueError(f"切片的起始位置({start_sec} s)晚于结束位置({end_sec} s)")
        if end_sec > self.duration:
            raise ValueError(f"切片结束位置({end_sec} s)越界(> {self.duration} s)")
        start_sample = int(round(start_sec * self._sample_rate))
        end_sample = int(round(end_sec * self._sample_rate))
        self._samples = self._samples[start_sample:end_sample]

    def random_subsegment(self, duration):
        """随机剪切指定长度的音频片段

        :param duration: 随机裁剪的片段长度，以秒为单位
        :type duration: float
        :raises ValueError: 如果片段长度大于原始段，则引发ValueError
        """
        if duration > self.duration:
            raise ValueError("裁剪的片段长度大于原始音频的长度")
        start_time = random.uniform(0.0, self.duration - duration)
        self.subsegment(start_time, start_time + duration)

    def reverb(self, reverb_file, allow_resample=True):
        """使音频片段混响

        :param reverb_file: 混响音频的路径
        :type reverb_file: str
        :param allow_resample: 指示是否允许在两个音频段具有不同的采样率时重采样
        :type allow_resample: bool
        :raises ValueError: 如果两个音频段之间的采样率不匹配，则引发ValueError
        """
        # 读取混响音频
        reverb_segment = AudioSegment.from_file(reverb_file)
        if allow_resample and self.sample_rate != reverb_segment.sample_rate:
            reverb_segment.resample(self.sample_rate)
        if self.sample_rate != reverb_segment.sample_rate:
            raise ValueError(f"音频的采样率为{self.sample_rate}，而混响的音频采样率{reverb_segment.sample_rate}")
        if reverb_segment.duration > self.duration:
            reverb_segment.random_subsegment(self.duration)
        reverb_samples = reverb_segment.samples
        reverb_samples = reverb_samples / np.sqrt(np.sum(reverb_samples ** 2))
        samples = signal.convolve(self.samples, reverb_samples, "full")
        samples = samples / (np.max(np.abs(samples)) + 1e-6)
        self._samples = samples[:self.num_samples]

    def reverb_and_normalize(self, reverb_file, allow_resample=True):
        """使音频片段混响，然后归一化

        :param reverb_file: 混响音频的路径
        :type reverb_file: str
        :param allow_resample: 指示是否允许在两个音频段具有不同的采样率时重采样
        :type allow_resample: bool
        :raises ValueError: 两个音频段之间的采样率不匹配
        """
        target_db = self.rms_db
        self.reverb(reverb_file, allow_resample=allow_resample)
        self.normalize(target_db)

    def add_noise(self, noise_file, snr_dB, max_gain_db=300.0, allow_resample=True):
        """以特定的信噪比添加给定的噪声段。如果噪声段比该噪声段长，则从该噪声段中采样匹配长度的随机子段。

        :param noise_file: 噪声音频的路径
        :type noise_file: str
        :param snr_dB: 信噪比，单位为分贝
        :type snr_dB: float
        :param max_gain_db: 最大允许的增益值，单位为分贝，这是为了防止在对全0信号进行归一化时出现Nan
        :type max_gain_db: float
        :param allow_resample: 指示是否允许在两个音频段具有不同的采样率时重采样
        :type allow_resample: bool
        :raises ValueError: 如果两个音频段之间的采样率不匹配，则引发ValueError
        """
        # 读取噪声音频
        noise_segment = AudioSegment.from_file(noise_file)
        if allow_resample and self.sample_rate != noise_segment.sample_rate:
            noise_segment.resample(self.sample_rate)
        if noise_segment.sample_rate != self.sample_rate:
            raise ValueError(f"噪声采样率({noise_segment.sample_rate} Hz)不等于基信号采样率({self.sample_rate} Hz)")
        if noise_segment.duration >= self.duration:
            noise_segment.random_subsegment(self.duration)
        else:
            # 如果噪声的长度小于基信号的长度，则将噪声的前面的部分填充噪声末尾补长
            num_samples = self.num_samples - noise_segment.num_samples
            noise_segment.pad((0, num_samples), mode='wrap')
        noise_gain_db = min(self.rms_db - noise_segment.rms_db - snr_dB, max_gain_db)
        noise_segment.gain_db(noise_gain_db)
        self.superimpose(noise_segment)

    # 裁剪音频
    def crop(self, duration, mode='eval'):
        """根据模式裁剪指定的音频长度，如果为'train'模式，则随机剪切，否则从末尾剪切

        :param duration: 裁剪的音频长度，以秒为单位
        :type duration: float
        :param mode: 裁剪的模型，'train'或'eval'
        :type mode: str
        """
        if self.duration > duration:
            if mode == 'train':
                self.random_subsegment(duration)
            else:
                self.subsegment(end_sec=duration)

    def vad(self, return_seconds=False, **kwargs):
        """使用VAD模型进行语音活动检测

        :param return_seconds: 指示是否返回秒数而不是样本索引
        :type return_seconds: bool
        :param kwargs: 传递给VAD模型的参数
        :type kwargs: dict
        :return: 语音活动时间戳列表
        :rtype: List[Dict]
        """
        if self.vad_model is None:
            from yeaudio.vad_model import VadModel
            self.vad_model = VadModel(**kwargs)
        speech_timestamps = self.vad_model(self.samples)[0]
        results = []
        if not return_seconds:
            for timestamp in speech_timestamps:
                result = {"start": timestamp[0] / 1000 * self.sample_rate,
                          "end": timestamp[1] / 1000 * self.sample_rate}
                results.append(result)
        else:
            for timestamp in speech_timestamps:
                result = {"start": timestamp[0] / 1000, "end": timestamp[1] / 1000}
                results.append(result)
        return results

    @property
    def samples(self):
        """
        :return: 返回音频样本
        :rtype: ndarray
        """
        return self._samples.copy()

    @property
    def sample_rate(self):
        """
        :return: 返回音频采样率
        :rtype: int
        """
        return self._sample_rate

    @property
    def num_samples(self):
        """
        :return: 返回样品数量
        :rtype: int
        """
        return self._samples.shape[0]

    @property
    def duration(self):
        """
        :return: 返回音频持续时间，以秒为单位
        :rtype: float
        """
        return self._samples.shape[0] / float(self._sample_rate)

    @property
    def rms_db(self):
        """
        :return: 返回以分贝为单位的音频均方根能量
        :rtype: float
        """
        # square root => multiply by 10 instead of 20 for dBs
        mean_square = np.mean(self._samples ** 2)
        if mean_square == 0:
            mean_square = 1
        return 10 * np.log10(mean_square)

    @staticmethod
    def _convert_samples_to_float32(samples):
        """把样本的类型转为float32.

        音频样本类型通常是整数或浮点数，整数将被缩放为float32类型的[- 1,1]。
        """
        float32_samples = samples.astype('float32')
        if samples.dtype in [np.int8, np.int16, np.int32, np.int64]:
            bits = np.iinfo(samples.dtype).bits
            float32_samples *= (1. / 2 ** (bits - 1))
        elif samples.dtype in [np.float16, np.float32, np.float64]:
            pass
        else:
            raise TypeError(f"Unsupported sample type: {samples.dtype}.")
        return float32_samples

    @staticmethod
    def _convert_samples_from_float32(samples, dtype):
        """Convert sample type from float32 to dtype.

        Audio sample type is usually integer or float-point. For integer
        type, float32 will be rescaled from [-1, 1] to the maximum range
        supported by the integer type.

        This is for writing a audio file.
        """
        dtype = np.dtype(dtype)
        output_samples = samples.copy()
        if dtype in [np.int8, np.int16, np.int32, np.int64]:
            bits = np.iinfo(dtype).bits
            output_samples *= (2 ** (bits - 1) / 1.)
            min_val = np.iinfo(dtype).min
            max_val = np.iinfo(dtype).max
            output_samples[output_samples > max_val] = max_val
            output_samples[output_samples < min_val] = min_val
        elif samples.dtype in [np.float16, np.float32, np.float64]:
            min_val = np.finfo(dtype).min
            max_val = np.finfo(dtype).max
            output_samples[output_samples > max_val] = max_val
            output_samples[output_samples < min_val] = min_val
        else:
            raise TypeError(f"Unsupported sample type: {samples.dtype}.")
        return output_samples.astype(dtype)
