from enum import Enum
from typing import Union

import numpy as np
from loguru import logger
from pydantic import BaseModel
from yeaudio.utils.silero_vad import load_silero_vad


# VAD状态
class VADState(Enum):
    QUIET = 1
    STARTING = 2
    SPEAKING = 3
    STOPPING = 4


# VAD参数
class VADParams(BaseModel):
    # 判断是否为静音的阈值
    threshold: float = 0.5
    # 开始检测的秒数
    start_secs: float = 0.1
    # 结束检测的秒数
    stop_secs: float = 0.1
    # 预测重置模型的秒数
    reset_model_secs: float = 5.0


class StreamingVAD(object):
    """流式语音活动（VAD）检测器

    :param sample_rate: 音频的采样率
    :type sample_rate: int
    :param num_channels: 音频的通道数
    :type num_channels: int
    :param params: VAD参数
    :type params: VADParams
    """

    def __init__(self, sample_rate: int = 16000, num_channels: int = 1, params: VADParams = VADParams()):
        assert sample_rate in [16000, 8000], "Only support 16kHz and 8kHz"
        assert num_channels == 1, "Only support mono audio"
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._params = params
        # 输入的音频numpy数据
        self._vad_data = []
        # 输入的音频字节数据
        self._vad_buffer = b""
        # 加载模型
        self._model = load_silero_vad()

        # 推理音频的numpy长度
        self.vad_frames = 512 if sample_rate == 16000 else 256
        # 推理音频的字节长度
        self.vad_frames_num_bytes = self.vad_frames * self._num_channels * 2
        vad_frames_per_sec = self.vad_frames / self._sample_rate
        # 判断为语音的帧数
        self._vad_start_frames = round(self._params.start_secs / vad_frames_per_sec)
        # 判断为静音的帧数
        self._vad_stop_frames = round(self._params.stop_secs / vad_frames_per_sec)
        self._vad_starting_count = 0
        self._vad_stopping_count = 0
        self._vad_state: VADState = VADState.QUIET
        # 已经推理的音频时间长度
        self._last_reset_time = 0

    def infer(self, buffer: Union[bytes, np.ndarray]) -> float:
        """推理语音活动（VAD）检测器

        :param buffer: 输入的音频，只能是vad_frames的长度或者vad_frames_num_bytes的字节长度
        :type buffer: Union[bytes, np.ndarray]
        :return: 是否为语音的概率
        :rtype: float
        """
        try:
            if isinstance(buffer, bytes):
                audio_int16 = np.frombuffer(buffer, np.int16)
                audio = np.frombuffer(audio_int16, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio = buffer.astype(np.float32)
            score = self._model(audio, self._sample_rate).item()
            return score
        except Exception as e:
            logger.error(f"Error while analyzing audio: {e}")
            return 0

    def __call__(self, buffer: Union[bytes, np.ndarray]) -> VADState:
        """推理语音活动（VAD）检测器

        :param buffer: 输入的音频，小于等于vad_frames的长度或者vad_frames_num_bytes的字节长度
        :type buffer: Union[bytes, np.ndarray]
        :return: 识别结果
        :rtype: VADState
        """
        if isinstance(buffer, bytes):
            assert len(buffer) <= self.vad_frames_num_bytes, \
                f"输入数据太长，需求长度不大于{self.vad_frames_num_bytes}，当前长度为{len(buffer)}"
            self._vad_buffer += buffer

            if len(self._vad_buffer) < self.vad_frames_num_bytes:
                return self._vad_state

            audio_frames = self._vad_buffer[:self.vad_frames_num_bytes]
            self._vad_buffer = self._vad_buffer[self.vad_frames_num_bytes:]
            audio_time = len(audio_frames) / (self._sample_rate * self._num_channels * 2)
        else:
            assert len(buffer) <= self.vad_frames, f"输入数据太长，需求长度不大于{self.vad_frames}，" \
                                                    f"当前长度为{buffer.shape[-1]}"
            if self._vad_data is None:
                self._vad_data = buffer
            else:
                self._vad_data = np.concatenate((self._vad_data, buffer))

            if len(self._vad_data) < self.vad_frames:
                return self._vad_state

            audio_frames = self._vad_data[:self.vad_frames]
            self._vad_data = self._vad_data[self.vad_frames:]
            audio_time = self._vad_data.shape[-1] / self._sample_rate
        # 推理识别
        speech_prob = self.infer(audio_frames)
        # 隔一段时间重置模型状态
        diff_time = self._last_reset_time + audio_time
        if diff_time >= self._params.reset_model_secs:
            self._model.reset_states()
            self._last_reset_time = 0

        if speech_prob >= self._params.threshold:
            match self._vad_state:
                case VADState.QUIET:
                    self._vad_state = VADState.STARTING
                    self._vad_starting_count = 1
                case VADState.STARTING:
                    self._vad_starting_count += 1
                case VADState.STOPPING:
                    self._vad_state = VADState.SPEAKING
                    self._vad_stopping_count = 0
        else:
            match self._vad_state:
                case VADState.STARTING:
                    self._vad_state = VADState.QUIET
                    self._vad_starting_count = 0
                case VADState.SPEAKING:
                    self._vad_state = VADState.STOPPING
                    self._vad_stopping_count = 1
                case VADState.STOPPING:
                    self._vad_stopping_count += 1
        # 统计语音的时间，判断是否已经开始说话
        if self._vad_state == VADState.STARTING and self._vad_starting_count >= self._vad_start_frames:
            self._vad_state = VADState.SPEAKING
            self._vad_starting_count = 0
        # 统计静音的时间，判断是否已经停止说话
        if self._vad_state == VADState.STOPPING and self._vad_stopping_count >= self._vad_stop_frames:
            self._vad_state = VADState.QUIET
            self._vad_stopping_count = 0

        return self._vad_state
