# This implementation is adapted from https://github.com/modelscope/FunASR
import os.path
from typing import List, Union, Tuple

import numpy as np
from loguru import logger

from yeaudio.utils.e2e_vad import E2EVadModel
from yeaudio.utils.frontend import WavFrontend, WavFrontendOnline
from yeaudio.utils.utils import ONNXRuntimeError, OrtInferSession, read_yaml


class VadModel:
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Deep-FSMN for Large Vocabulary Continuous Speech Recognition
    https://arxiv.org/abs/1803.05030
    """

    def __init__(
            self,
            batch_size: int = 1,
            device_id: Union[str, int] = "-1",
            quantize: bool = False,
            intra_op_num_threads: int = 4,
            max_end_sil: int = None,
    ):
        """VAD（语音检测活动检测）模型

        Args:
            batch_size (int, optional): 批处理大小，默认为1。
            device_id (Union[str, int], optional): 设备ID，用于指定模型运行的设备，默认为"-1"表示使用CPU。如果指定为GPU，则为GPU的ID。
            quantize (bool, optional): 是否使用量化模型，默认为False。
            intra_op_num_threads (int, optional): ONNX Runtime的线程数，默认为4。
            max_end_sil (int, optional): 最大静默结束时间，如果未指定，则使用模型配置中的默认值。
        """
        model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'utils/vad_models/')
        model_file = os.path.join(model_dir, "model.onnx")
        if quantize:
            model_file = os.path.join(model_dir, "model_quant.onnx")
        config_file = os.path.join(model_dir, "config.yaml")
        cmvn_file = os.path.join(model_dir, "am.mvn")
        config = read_yaml(config_file)

        self.frontend = WavFrontend(cmvn_file=cmvn_file, **config["frontend_conf"])
        self.sample_rate = self.frontend.opts.frame_opts.samp_freq
        self.ort_infer = OrtInferSession(model_file, device_id, intra_op_num_threads=intra_op_num_threads)
        self.batch_size = batch_size
        self.vad_scorer = E2EVadModel(config["model_conf"])
        self.max_end_sil = (max_end_sil if max_end_sil is not None else config["model_conf"]["max_end_silence_time"])
        self.encoder_conf = config["encoder_conf"]

    def prepare_cache(self, in_cache: list = []):
        if len(in_cache) > 0:
            return in_cache
        fsmn_layers = self.encoder_conf["fsmn_layers"]
        proj_dim = self.encoder_conf["proj_dim"]
        lorder = self.encoder_conf["lorder"]
        for i in range(fsmn_layers):
            cache = np.zeros((1, proj_dim, lorder - 1, 1)).astype(np.float32)
            in_cache.append(cache)
        return in_cache

    def __call__(self, audio_in: Union[np.ndarray, List[np.ndarray]], **kwargs) -> List:
        """
        调用对象实例
        Args:
            audio_in (Union[np.ndarray, List[np.ndarray]]): 输入音频数据，可以是单个numpy数组或numpy数组列表，采样率为16000
            **kwargs: 其他关键字参数
        Returns:
            List: 返回结构为[[开始, 结束],[开始, 结束]...]，单位毫秒
        """
        if not isinstance(audio_in, list):
            waveform_list = [audio_in]
        else:
            waveform_list = audio_in
        waveform_nums = len(waveform_list)

        segments = [[]] * self.batch_size
        for beg_idx in range(0, waveform_nums, self.batch_size):

            end_idx = min(waveform_nums, beg_idx + self.batch_size)
            waveform = waveform_list[beg_idx:end_idx]
            feats, feats_len = self.extract_feat(waveform)
            waveform = np.array(waveform)
            param_dict = kwargs.get("param_dict", dict())
            in_cache = param_dict.get("in_cache", list())
            in_cache = self.prepare_cache(in_cache)
            try:
                t_offset = 0
                step = int(min(feats_len.max(), 6000))
                for t_offset in range(0, int(feats_len), min(step, feats_len - t_offset)):
                    if t_offset + step >= feats_len - 1:
                        step = feats_len - t_offset
                        is_final = True
                    else:
                        is_final = False
                    feats_package = feats[:, t_offset: int(t_offset + step), :]
                    waveform_package = \
                        waveform[:, t_offset * 160: min(waveform.shape[-1], (int(t_offset + step) - 1) * 160 + 400), ]

                    inputs = [feats_package]
                    inputs.extend(in_cache)
                    scores, out_caches = self.infer(inputs)
                    in_cache = out_caches
                    segments_part = self.vad_scorer(
                        scores,
                        waveform_package,
                        is_final=is_final,
                        max_end_sil=self.max_end_sil,
                        online=False,
                    )

                    if segments_part:
                        for batch_num in range(0, self.batch_size):
                            segments[batch_num] += segments_part[batch_num]

            except ONNXRuntimeError:
                logger.exception("input wav is silence or noise")
                segments = ""

        return segments

    def extract_feat(self, waveform_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        feats, feats_len = [], []
        for waveform in waveform_list:
            speech, _ = self.frontend.fbank(waveform)
            feat, feat_len = self.frontend.lfr_cmvn(speech)
            feats.append(feat)
            feats_len.append(feat_len)

        feats = self.pad_feats(feats, np.max(feats_len))
        feats_len = np.array(feats_len).astype(np.int32)
        return feats, feats_len

    @staticmethod
    def pad_feats(feats: List[np.ndarray], max_feat_len: int) -> np.ndarray:
        def pad_feat(feat: np.ndarray, cur_len: int) -> np.ndarray:
            pad_width = ((0, max_feat_len - cur_len), (0, 0))
            return np.pad(feat, pad_width, "constant", constant_values=0)

        feat_res = [pad_feat(feat, feat.shape[0]) for feat in feats]
        feats = np.array(feat_res).astype(np.float32)
        return feats

    def infer(self, feats: List) -> Tuple[np.ndarray, np.ndarray]:
        outputs = self.ort_infer(feats)
        scores, out_caches = outputs[0], outputs[1:]
        return scores, out_caches


class VadOnlineModel:
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Deep-FSMN for Large Vocabulary Continuous Speech Recognition
    https://arxiv.org/abs/1803.05030
    """

    def __init__(self,
                 batch_size: int = 1,
                 device_id: Union[str, int] = "-1",
                 quantize: bool = True,
                 intra_op_num_threads: int = 4,
                 max_end_sil: int = None):
        """VAD（语音检测活动检测）模型，流式识别

        Args:
            batch_size (int, optional): 批处理大小，默认为1。
            device_id (Union[str, int], optional): 设备ID，用于指定模型运行的设备，默认为"-1"表示使用CPU。如果指定为GPU，则为GPU的ID。
            quantize (bool, optional): 是否使用量化模型，默认为False。
            intra_op_num_threads (int, optional): ONNX Runtime的线程数，默认为4。
            max_end_sil (int, optional): 最大静默结束时间，如果未指定，则使用模型配置中的默认值。
        """
        model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'utils/vad_models/')
        model_file = os.path.join(model_dir, "model.onnx")
        if quantize:
            model_file = os.path.join(model_dir, "model_quant.onnx")
        config_file = os.path.join(model_dir, "config.yaml")
        cmvn_file = os.path.join(model_dir, "am.mvn")
        config = read_yaml(config_file)

        self.frontend = WavFrontendOnline(cmvn_file=cmvn_file, **config["frontend_conf"])
        self.sample_rate = self.frontend.opts.frame_opts.samp_freq
        self.ort_infer = OrtInferSession(model_file, device_id, intra_op_num_threads=intra_op_num_threads)
        self.batch_size = batch_size
        self.vad_scorer = E2EVadModel(config["model_conf"])
        self.max_end_sil = (max_end_sil if max_end_sil is not None else config["model_conf"]["max_end_silence_time"])
        self.encoder_conf = config["encoder_conf"]

    def prepare_cache(self, in_cache: list = []):
        if len(in_cache) > 0:
            return in_cache
        fsmn_layers = self.encoder_conf["fsmn_layers"]
        proj_dim = self.encoder_conf["proj_dim"]
        lorder = self.encoder_conf["lorder"]
        for i in range(fsmn_layers):
            cache = np.zeros((1, proj_dim, lorder - 1, 1)).astype(np.float32)
            in_cache.append(cache)
        return in_cache

    def __call__(self, audio_in: np.ndarray, **kwargs) -> List:
        """
        调用对象实例
        Args:
            audio_in (np.ndarray): 输入音频数据，采样率为16000
            **kwargs: 其他关键字参数
        Returns:
            List: 返回结构为[[开始, 结束],[开始, 结束]...]，如果是-1，则包含该位置，如果为[]，没有检测到活动事件，单位毫秒
        """
        waveforms = np.expand_dims(audio_in, axis=0)

        param_dict = kwargs.get("param_dict", dict())
        is_final = param_dict.get("is_final", False)
        feats, feats_len = self.extract_feat(waveforms, is_final)
        segments = []
        if feats.size != 0:
            in_cache = param_dict.get("in_cache", list())
            in_cache = self.prepare_cache(in_cache)
            try:
                inputs = [feats]
                inputs.extend(in_cache)
                scores, out_caches = self.infer(inputs)
                param_dict["in_cache"] = out_caches
                waveforms = self.frontend.get_waveforms()
                segments = self.vad_scorer(
                    scores, waveforms, is_final=is_final, max_end_sil=self.max_end_sil, online=True)
                if len(segments) > 0:
                    segments = segments[0]
            except ONNXRuntimeError:
                logger.exception("input wav is silence or noise")
                segments = []
        return segments

    def extract_feat(self, waveforms: np.ndarray, is_final: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        waveforms_lens = np.zeros(waveforms.shape[0]).astype(np.int32)
        for idx, waveform in enumerate(waveforms):
            waveforms_lens[idx] = waveform.shape[-1]
        feats, feats_len = self.frontend.extract_fbank(waveforms, waveforms_lens, is_final)
        return feats.astype(np.float32), feats_len.astype(np.int32)

    @staticmethod
    def pad_feats(feats: List[np.ndarray], max_feat_len: int) -> np.ndarray:
        def pad_feat(feat: np.ndarray, cur_len: int) -> np.ndarray:
            pad_width = ((0, max_feat_len - cur_len), (0, 0))
            return np.pad(feat, pad_width, "constant", constant_values=0)

        feat_res = [pad_feat(feat, feat.shape[0]) for feat in feats]
        feats = np.array(feat_res).astype(np.float32)
        return feats

    def infer(self, feats: List) -> Tuple[np.ndarray, np.ndarray]:

        outputs = self.ort_infer(feats)
        scores, out_caches = outputs[0], outputs[1:]
        return scores, out_caches
