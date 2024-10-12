# 夜雨飘零音频工具

这款Python音频处理工具功能强大，支持读取多种格式的音频文件。它不仅能够对音频进行裁剪、添加混响、添加噪声等多种处理操作，还广泛应用于语音识别、语音合成、声音分类以及声纹识别等多个项目领域。

# 安装

使用pip安装。

```shell
pip install yeaudio -U -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**（推荐）** 使用源码安装。

```shell
git clone https://github.com/yeyupiaoling/YeAudio.git
cd YeAudio
pip install . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

# 快速使用

读取普通音频：
```python
from yeaudio.audio import AudioSegment

audio_segment = AudioSegment.from_file('data/test.wav')
print(f'音频长度：{audio_segment.duration}')
print(f'音频采样率：{audio_segment.sample_rate}')
print(f'音频数据：{audio_segment.samples}')
```

读取视频中的音频：
```python
from yeaudio.audio import AudioSegment

audio_segment = AudioSegment.from_file('data/test.mp4')
print(f'音频长度：{audio_segment.duration}')
print(f'音频采样率：{audio_segment.sample_rate}')
print(f'音频数据：{audio_segment.samples}')
```

# API文档

 - [AudioSegment](#AudioSegment)
 - [VadModel](#VadModel)
 - [VadOnlineModel](#VadOnlineModel)
 - [SpeedPerturbAugmentor](#SpeedPerturbAugmentor)
 - [VolumePerturbAugmentor](#VolumePerturbAugmentor)
 - [ShiftPerturbAugmentor](#ShiftPerturbAugmentor)
 - [ResampleAugmentor](#ResampleAugmentor)
 - [NoisePerturbAugmentor](#NoisePerturbAugmentor)
 - [ReverbPerturbAugmentor](#ReverbPerturbAugmentor)
 - [SpecAugmentor](#SpecAugmentor)
 - [SpecSubAugmentor](#SpecSubAugmentor)

## AudioSegment

基础音频工具，支持读取多种格式的音频文件，已经各种基础操作，如裁剪、添加混响、添加噪声等。

<br/>

> **def `__init__`(self, samples, sample_rate):**

创建单通道音频片段实例

**参数：**

 - **samples（ndarray.float32）：** 频数据，维度为[num_samples x num_channels]
 - **sample_rate（int）：** 音频的采样率

**示例代码：**

```python
import soundfile
from yeaudio.audio import AudioSegment

samples, sample_rate = soundfile.read("data/test.wav")
audio_segment = AudioSegment(samples, sample_rate)
print(audio_segment.samples)
```
<br/>

---

> **def `__eq__`(self, other):**

返回两个对象是否相等

**参数：**

 - **other（AudioSegment）：** 比较的另一个音频片段实例

**示例代码：**

```python
from yeaudio.audio import AudioSegment

audio_segment1 = AudioSegment.from_file("data/test.wav")
audio_segment2 = AudioSegment.from_file("data/test.wav")
print(audio_segment1 == audio_segment2)
```
<br/>

---

> **def `__ne__`(self, other):**

返回两个实例是否不相等

**参数：**

 - **other（AudioSegment）：** 比较的另一个音频片段实例

**示例代码：**

```python
from yeaudio.audio import AudioSegment

audio_segment1 = AudioSegment.from_file("data/test.wav")
audio_segment2 = AudioSegment.from_file("data/test.wav")
print(audio_segment1 != audio_segment2)
```
<br/>

---

> **def `__str__`(self):**

返回该音频的信息

**示例代码：**

```python
from yeaudio.audio import AudioSegment

audio_segment = AudioSegment.from_file("data/test.wav")
print(str(audio_segment))
```
<br/>

---

> **@classmethod**<br/>
> **def from_file(cls, file):**

从音频文件创建音频段，支持wav、mp3、mp4等多种音频格式

**参数：**

 - **file（str|BufferedReader）：** 件路径，或者文件对象

**返回：**

 - `AudioSegment`：音频片段实例

**示例代码：**

```python
from yeaudio.audio import AudioSegment

audio_segment = AudioSegment.from_file('data/test.wav')
print(audio_segment.samples)
```
<br/>

---

> **@classmethod**<br/>
> **def slice_from_file(cls, file, start=None, end=None):**

只加载一小段音频，而不需要将整个文件加载到内存中，这是非常浪费的。

**参数：**

 - **file（str|file）：** 输入音频文件路径或文件对象
 - **start（float）：** 开始时间，单位为秒。如果start是负的，则它从末尾开始计算。如果没有提供，这个函数将从最开始读取。
 - **end（float）：** 结束时间，单位为秒。如果end是负的，则它从末尾开始计算。如果没有提供，默认的行为是读取到文件的末尾。

**返回：**

 - `AudioSegment`：AudioSegment输入音频文件的指定片的实例

**异常：**

 - `ValueError`：如果开始或结束的设定不正确，则会抛出ValueError异常

**示例代码：**

```python
from yeaudio.audio import AudioSegment

audio_segment = AudioSegment.slice_from_file('data/test.wav', start=1, end=2)
print(audio_segment.samples)
```
<br/>

---

> **@classmethod**<br/>
> **def from_bytes(cls, data):**

从wav格式的音频字节创建音频段

**参数：**

 - **data（bytes）：** 包含音频样本的字节

**返回：**

 - `AudioSegment`：音频片段实例

**示例代码：**

```python
from yeaudio.audio import AudioSegment

with open('data/test.wav', 'rb') as f:
    data = f.read()
    audio_segment = AudioSegment.from_bytes(data)
    print(audio_segment.samples)
```
<br/>

---

> **@classmethod**<br/>
> **def from_pcm_bytes(cls, data, channels=1, samp_width=2, sample_rate=16000):**

从包含无格式PCM音频的字节创建音频

**参数：**

 - **data（bytes）：** 包含音频样本的字节
 - **channels（int）：** 音频的通道数
 - **samp_width（int）：** 频采样的宽度，如np.int16为2
 - **sample_rate（int）：** 音频样本采样率

**返回：**

 - `AudioSegment`：音频片段实例

**示例代码：**

```python
from yeaudio.audio import AudioSegment

with open('data/test.wav', 'rb') as f:
    data = f.read()
    audio_segment = AudioSegment.from_pcm_bytes(data[44:], channels=1, samp_width=2, sample_rate=16000)
    print(audio_segment.samples)
```
<br/>

---

> **@classmethod**<br/>
> **def from_ndarray(cls, data, sample_rate=16000):**

从numpy.ndarray创建音频段

**参数：**

 - **data（bytes）：** numpy.ndarray类型的音频数据
 - **sample_rate（int）：** 音频样本采样率

**返回：**

 - `AudioSegment`：音频片段实例

**示例代码：**

```python
import soundfile

from yeaudio.audio import AudioSegment

samples, sample_rate = soundfile.read('data/test.wav')
audio_segment = AudioSegment.from_ndarray(samples, sample_rate=16000)
print(audio_segment.samples)
```
<br/>

---

> **@classmethod**<br/>
> **def concatenate(cls, \*segments):**

将任意数量的音频片段连接在一起

**参数：**

 - **segments（AudioSegment）：** 输入音频片段被连接

**返回：**

 - `AudioSegment`：音频片段实例

**异常：**

 - `ValueError`：如果音频实例列表为空或者采样率不一致，则会抛出ValueError异常
 - `TypeError`：如果输入的片段类型不一致，则会抛出TypeError异常

**示例代码：**

```python
from yeaudio.audio import AudioSegment

audio_segment1 = AudioSegment.from_file('data/test.wav')
audio_segment2 = AudioSegment.from_file('data/test.wav')
audio_segment = AudioSegment.concatenate(audio_segment1, audio_segment2)
print(audio_segment.samples)
```
<br/>

---

> **@classmethod**<br/>
> **def make_silence(cls, duration, sample_rate):**

创建给定持续时间和采样率的静音音频段

**参数：**

 - **duration（float）：** 静音的时间，以秒为单位
 - **sample_rate（int）：** 音频采样率

**返回：**

 - `AudioSegment`：给定持续时间的静音AudioSegment实例

**示例代码：**

```python
from yeaudio.audio import AudioSegment

audio_segment = AudioSegment.make_silence(duration=10, sample_rate=16000)
print(audio_segment.samples)
```
<br/>

---

> **def to_wav_file(self, filepath, dtype='float32'):**

保存音频段到磁盘为wav文件

**参数：**

 - **filepath（str|file）：** WAV文件路径或文件对象，以保存音频段
 - **dtype（str）：** 音频数据类型，可选: 'int16', 'int32', 'float32', 'float64'

**异常：**

 - `TypeError`：如果类型不支持，则会抛出TypeError异常

**示例代码：**

```python
from yeaudio.audio import AudioSegment

audio_segment = AudioSegment.from_file("data/test.wav")
audio_segment.to_wav_file("output.wav")
```
<br/>

---

> **def superimpose(self, other):**

将另一个段的样本添加到这个段的样本中(以样本方式添加，而不是段连接)。

**参数：**

 - **other（AudioSegments）：** WAV文件路径或文件对象，以保存音频段
 - **dtype（str）：** 音频数据类型，可选: 'int16', 'int32', 'float32', 'float64'

**异常：**

 - `ValueError`：如果两段音频采样率或者长度不一致，则会抛出ValueError异常
 - `TypeError`：如果两个片段的类型不匹配，则会抛出TypeError异常

**示例代码：**

```python
from yeaudio.audio import AudioSegment

audio_segment = AudioSegment.from_file("data/test.wav")
other_segment = AudioSegment.from_file("data/test.wav")
audio_segment.superimpose(other_segment)
```
<br/>

---

> **def to_bytes(self, dtype='float32'):**

创建音频内容的字节

**参数：**

 - **dtype（str）：** 导出样本的数据类型。可选: 'int16', 'int32', 'float32', 'float64'

**返回：**

 - `bytes`：音频内容的字节

**示例代码：**

```python
from yeaudio.audio import AudioSegment

audio_segment = AudioSegment.from_file("data/test.wav")
print(audio_segment.to_bytes())
```
<br/>

---

> **def to_pcm_bytes(self):**

创建pcm格式的字节

**返回：**

 - `bytes`：pcm格式的字节

**示例代码：**

```python
from yeaudio.audio import AudioSegment

audio_segment = AudioSegment.from_file("data/test.wav")
print(audio_segment.to_pcm_bytes())
```
<br/>

---

> **def to(self, dtype='int16'):**

类型转换

**参数：**

 - **dtype（str）：** 导出样本的数据类型。可选: 'int16', 'int32', 'float32', 'float64'

**返回：**

 - `str`：转换后的数据

**示例代码：**

```python
from yeaudio.audio import AudioSegment

audio_segment = AudioSegment.from_file("data/test.wav")
print(audio_segment.to(dtype='int16'))
```
<br/>

---

> **def gain_db(self, gain):**

对音频施加分贝增益。

**参数：**

 - **gain（float|1darray）：** 用于样品的分贝增益

**示例代码：**

```python
from yeaudio.audio import AudioSegment

audio_segment = AudioSegment.from_file("data/test.wav")
audio_segment.gain_db(gain=-20)
print(audio_segment.samples)
```
<br/>

---

> **def change_speed(self, speed_rate):**

通过线性插值改变音频速度。

**参数：**

 - **speed_rate（float）：** 修改的音频速率: speed_rate > 1.0, 加快音频速度; speed_rate = 1.0, 音频速度不变; speed_rate < 1.0, 减慢音频速度; speed_rate <= 0.0, 错误数值.

**异常：**

 - `ValueError`：如果速度速率小于或等于0，则引发ValueError

**示例代码：**

```python
from yeaudio.audio import AudioSegment

audio_segment = AudioSegment.from_file("data/test.wav")
audio_segment.change_speed(speed_rate=1.2)
print(audio_segment.samples)
```
<br/>

---

> **def normalize(self, target_db=-20, max_gain_db=300.0):**

将音频归一化，使其具有所需的有效值(以分贝为单位)。

**参数：**

 - **target_db（float）：** 目标均方根值，单位为分贝。这个值应该小于0.0，因为0.0是全尺寸音频。
 - **max_gain_db（float）：** 最大允许的增益值，单位为分贝，这是为了防止在对全0信号进行归一化时出现Nan值。

**异常：**

 - `ValueError`：如果所需的增益大于max_gain_db，则引发ValueError

**示例代码：**

```python
from yeaudio.audio import AudioSegment

audio_segment = AudioSegment.from_file("data/test.wav")
audio_segment.normalize(target_db=-20)
print(audio_segment.samples)
```
<br/>

---

> **def resample(self, target_sample_rate, filter='kaiser_best'):**

按目标采样率重新采样音频。

**参数：**

 - **target_sample_rate（int）：** 目标均方根值，单位为分贝。这个值应该小于0.0，因为0.0是全尺寸音频。
 - **filter（str）：** 使用的重采样滤波器，支持'kaiser_best'、'kaiser_fast'

**示例代码：**

```python
from yeaudio.audio import AudioSegment

audio_segment = AudioSegment.from_file("data/test.wav")
audio_segment.resample(target_sample_rate=8000)
print(audio_segment.samples)
```
<br/>

---

> **def pad_silence(self, duration, sides='both'):**

在这个音频样本上加一段静音。

**参数：**

 - **duration（float）：** 静默段的持续时间(以秒为单位)
 - **sides（str）：** 添加的位置: 'beginning' - 在开始位置前增加静音段; 'end' - 在结束位置增加静音段; 'both' - 在开始和结束位置都增加静音段.。

**异常：**

 - `ValueError`：如果sides的值不是beginning、end或both，则引发ValueError

**示例代码：**

```python
from yeaudio.audio import AudioSegment

audio_segment = AudioSegment.from_file("data/test.wav")
audio_segment.pad_silence(duration=2, sides='end')
print(audio_segment.samples)
```
<br/>

---

> **def pad(self, pad_width, mode='wrap', \*\*kwargs):**

在这个音频样本上加一段音频，等同numpy.pad。

**参数：**

 - **pad_width（sequence|array_like|int）：** 填充宽度
 - **sides（str|function|optional）：** 填充模式

**示例代码：**

```python
from yeaudio.audio import AudioSegment

audio_segment = AudioSegment.from_file("data/test.wav")
audio_segment.pad(pad_width=(0, 16000 * 2), mode='wrap')
print(audio_segment.samples)
```
<br/>

---

> **def shift(self, shift_ms):**

音频偏移。如果shift_ms为正，则随时间提前移位;如果为负，则随时间延迟移位。填补静音以保持持续时间不变。

**参数：**

 - **shift_ms（float）：** 偏移时间。如果是正的，随时间前进；如果负，延时移位。

**异常：**

 - `ValueError`：如果shift_ms的绝对值大于音频持续时间，则引发ValueError

**示例代码：**

```python
from yeaudio.audio import AudioSegment

audio_segment = AudioSegment.from_file("data/test.wav")
audio_segment.shift(shift_ms=1000)
print(audio_segment.samples)
```
<br/>

---

> **def subsegment(self, start_sec=None, end_sec=None):**

在给定的边界之间切割音频片段。

**参数：**

 - **start_sec（float）：** 开始裁剪的位置，以秒为单位，默认为0。
 - **end_sec（float）：** 结束裁剪的位置，以秒为单位，默认为音频长度。

**异常：**

 - `ValueError`：如果start_sec或end_sec的值越界，则引发ValueError

**示例代码：**

```python
from yeaudio.audio import AudioSegment

audio_segment = AudioSegment.from_file("data/test.wav")
audio_segment.subsegment(start_sec=1, end_sec=3)
print(audio_segment.samples)
```
<br/>

---

> **def random_subsegment(self, duration):**

随机剪切指定长度的音频片段。

**参数：**

 - **duration（float）：** 随机裁剪的片段长度，以秒为单位

**异常：**

 - `ValueError`：如果片段长度大于原始段，则引发ValueError

**示例代码：**

```python
from yeaudio.audio import AudioSegment

audio_segment = AudioSegment.from_file("data/test.wav")
audio_segment.random_subsegment(duration=2)
print(audio_segment.samples)
```
<br/>

---

> **def reverb(self, reverb_file, allow_resample=True):**

使音频片段混响。

**参数：**

 - **reverb_file（str）：** 混响音频的路径
 - **allow_resample（bool）：** 指示是否允许在两个音频段具有不同的采样率时重采样

**异常：**

 - `ValueError`：如果两个音频段之间的采样率不匹配，则引发ValueError

**示例代码：**

```python
from yeaudio.audio import AudioSegment

audio_segment = AudioSegment.from_file("data/test.wav")
audio_segment.reverb(reverb_file='data/reverb.wav')
print(audio_segment.samples)
```
<br/>

---

> **def reverb_and_normalize(self, reverb_file, allow_resample=True):**

使音频片段混响，然后归一化。

**参数：**

 - **reverb_file（str）：** 混响音频的路径
 - **allow_resample（bool）：** 指示是否允许在两个音频段具有不同的采样率时重采样

**异常：**

 - `ValueError`：如果两个音频段之间的采样率不匹配，则引发ValueError

**示例代码：**

```python
from yeaudio.audio import AudioSegment

audio_segment = AudioSegment.from_file("data/test.wav")
audio_segment.reverb_and_normalize(reverb_file='data/reverb.wav')
print(audio_segment.samples)
```
<br/>

---

> **def add_noise(self, noise_file, snr_dB, max_gain_db=300.0, allow_resample=True):**

以特定的信噪比添加给定的噪声段。如果噪声段比该噪声段长，则从该噪声段中采样匹配长度的随机子段。

**参数：**

 - **noise_file（str）：** 噪声音频的路径
 - **snr_dB（float）：** 信噪比，单位为分贝
 - **max_gain_db（float）：** 最大允许的增益值，单位为分贝，这是为了防止在对全0信号进行归一化时出现Nan
 - **allow_resample（bool）：** 指示是否允许在两个音频段具有不同的采样率时重采样

**异常：**

 - `ValueError`：如果两个音频段之间的采样率不匹配，则引发ValueError

**示例代码：**

```python
from yeaudio.audio import AudioSegment

audio_segment = AudioSegment.from_file("data/test.wav")
audio_segment.add_noise(noise_file='data/noise.wav', snr_dB=10)
print(audio_segment.samples)
```
<br/>

---

> **def crop(self, duration, mode='eval'):**

根据模式裁剪指定的音频长度，如果为'train'模式，则随机剪切，否则从末尾剪切。

**参数：**

 - **duration（float）：** 裁剪的音频长度，以秒为单位
 - **mode（str）：** 裁剪的模型，'train'或'eval'

**异常：**

 - `ValueError`：如果两个音频段之间的采样率不匹配，则引发ValueError

**示例代码：**

```python
from yeaudio.audio import AudioSegment

audio_segment = AudioSegment.from_file("data/test.wav")
audio_segment.crop(duration=3, mode='train')
print(audio_segment.samples)
```
<br/>

---

> **def vad(self, return_seconds=False, \*\*kwargs):**

创建给定持续时间和采样率的静音音频段。

**参数：**

 - **return_seconds（bool）：** 指示是否返回秒数而不是样本索引
 - **kwargs（dict）：** 传递给Silero VAD模型的参数

**返回：**

 - `List[Dict]`：语音活动时间戳列表

**示例代码：**

```python
from yeaudio.audio import AudioSegment

audio_segment = AudioSegment.from_file("data/test_long.wav")
speech_timestamps = audio_segment.vad(return_seconds=True)
for speech_timestamp in speech_timestamps:
    print(speech_timestamp)
```
<br/>

---

> **@property**<br/>
> **def samples(self):**

返回音频样本

**返回：**

 - `float`：返回音频样本

**示例代码：**

```python
from yeaudio.audio import AudioSegment

audio_segment = AudioSegment.from_file('data/test.wav')
print(audio_segment.samples)
```
<br/>

---

> **@property**<br/>
> **def sample_rate(self):**

返回音频采样率

**返回：**

 - `int`：返回音频采样率

**示例代码：**

```python
from yeaudio.audio import AudioSegment

audio_segment = AudioSegment.from_file('data/test.wav')
print(audio_segment.sample_rate)
```
<br/>

---

> **@property**<br/>
> **def num_samples(self):**

返回样品数量

**返回：**

 - `int`：返回样品数量

**示例代码：**

```python
from yeaudio.audio import AudioSegment

audio_segment = AudioSegment.from_file('data/test.wav')
print(audio_segment.num_samples)
```
<br/>

---

> **@property**<br/>
> **def duration(self):**

返回音频持续时间，以秒为单位

**返回：**

 - `float`：返回音频持续时间，以秒为单位

**示例代码：**

```python
from yeaudio.audio import AudioSegment

audio_segment = AudioSegment.from_file('data/test.wav')
print(audio_segment.duration)
```
<br/>

---

> **@property**<br/>
> **def rms_db(self):**

返回以分贝为单位的音频均方根能量

**返回：**

 - `float`：返回以分贝为单位的音频均方根能量

**示例代码：**

```python
from yeaudio.audio import AudioSegment

audio_segment = AudioSegment.from_file('data/test.wav')
print(audio_segment.rms_db)
```
<br/>

## VadModel

语音活动检测模型，非流式

> **def `__init__`(self,
                   batch_size: int = 1,
                   device_id: Union[str, int] = "-1",
                   quantize: bool = True,
                   intra_op_num_threads: int = 4,
                   max_end_sil: int = None):**

**参数：**

 - **batch_size (int, optional):** 批处理大小，默认为1。
 - **device_id (Union[str, int], optional):** 设备ID，用于指定模型运行的设备，默认为"-1"表示使用CPU。如果指定为GPU，则为GPU的ID。
 - **quantize (bool, optional):** 是否使用量化模型，默认为False。
 - **intra_op_num_threads (int, optional):** ONNX Runtime的线程数，默认为4。
 - **max_end_sil (int, optional):** 最大静默结束时间，如果未指定，则使用模型配置中的默认值。

> **def `__call__`(self, audio_in: Union[np.ndarray, List[np.ndarray]]) -> List:**

**参数：**

 - **audio_in (Union[np.ndarray, List[np.ndarray]]):** 输入音频数据，可以是单个numpy数组或numpy数组列表，采样率为16000

**返回：**

 - List: 返回结构为[[开始, 结束],[开始, 结束]...]，如果是-1，则包含该位置，如果为[]，没有检测到活动事件，单位毫秒


**示例代码：**

```python
from yeaudio.audio import AudioSegment
from yeaudio.vad_model import VadModel

vad_model = VadModel()
audio_segment = AudioSegment.from_file("data/test_long.wav")
audio_segment.resample(target_sample_rate=vad_model.sample_rate)
samples = audio_segment.samples

speech_timestamps = vad_model(samples)
for speech_timestamp in speech_timestamps:
    print(speech_timestamp)
```


## VadOnlineModel


语音活动检测模型，在线，或者叫流式

> **def `__init__`(self,
                   batch_size: int = 1,
                   device_id: Union[str, int] = "-1",
                   quantize: bool = True,
                   intra_op_num_threads: int = 4,
                   max_end_sil: int = None):**

**参数：**

 - **batch_size (int, optional):** 批处理大小，默认为1。
 - **device_id (Union[str, int], optional):** 设备ID，用于指定模型运行的设备，默认为"-1"表示使用CPU。如果指定为GPU，则为GPU的ID。
 - **quantize (bool, optional):** 是否使用量化模型，默认为False。
 - **intra_op_num_threads (int, optional):** ONNX Runtime的线程数，默认为4。
 - **max_end_sil (int, optional):** 最大静默结束时间，如果未指定，则使用模型配置中的默认值。

> **def `__call__`(self, audio_in: np.ndarray) -> List:**

**参数：**

 - **audio_in (np.ndarray):** 输入音频数据，采样率为16000

**返回：**

 - List: 返回结构为[[开始, 结束],[开始, 结束]...]，如果是-1，则包含该位置，如果为[]，没有检测到活动事件，单位毫秒


**示例代码：**

```python
from yeaudio.audio import AudioSegment
from yeaudio.vad_model import VadOnlineModel

vad_model = VadOnlineModel()

audio_segment = AudioSegment.from_file('data/test_long.wav')
audio_segment.resample(target_sample_rate=vad_model.sample_rate)
samples = audio_segment.samples

speech_length = len(samples)
step = 16000
param_dict = {"in_cache": []}
for sample_offset in range(0, speech_length, step):
    is_final = True if sample_offset + step >= speech_length - 1 else False
    data = samples[sample_offset:sample_offset + step]
    param_dict["is_final"] = is_final
    segments_result = vad_model(audio_in=data, param_dict=param_dict)
    if len(segments_result) > 0:
        print("segments_result", segments_result)

```

## SpeedPerturbAugmentor

随机语速扰动的音频数据增强器


> **def `__init__`(self, prob=1.0, speed_perturb_3_class=False, num_speakers=None):**

**参数：**

 - **prob（float）：** 数据增强概率
 - **speed_perturb_3_class（bool）：** 是否使用语速三类语速增强，只在声纹识别项目上使用
 - **num_speakers（int）：** 说话人数量，只在声纹识别项目上使用

> **def `__call__`(self, audio_segment: AudioSegment, spk_id: int = None) -> AudioSegment or \[AudioSegment, int]:**

**参数：**

 - **audio_segment：** AudioSegment实例


## VolumePerturbAugmentor

随机音量扰动的音频数据增强器


> **def `__init__`(self, prob=0.0, min_gain_dBFS=-15, max_gain_dBFS=15):**

**参数：**

 - **prob（float）：** 数据增强概率
 - **min_gain_dBFS（int）：** 最小音量，单位为分贝。
 - **max_gain_dBFS（int）：** 最大音量，单位为分贝。

> **def `__call__`(self, audio_segment: AudioSegment) -> AudioSegment:**

**参数：**

 - **audio_segment：** AudioSegment实例


## ShiftPerturbAugmentor

添加随机位移扰动的音频数增强器


> **def `__init__`(self, prob=0.0, min_shift_ms=-15, max_shift_ms=15):**

**参数：**

 - **prob（float）：** 数据增强概率
 - **min_shift_ms（int）：** 最小偏移，单位为毫秒。
 - **max_shift_ms（int）：** 最大偏移，单位为毫秒。

> **def `__call__`(self, audio_segment: AudioSegment) -> AudioSegment:**

**参数：**

 - **audio_segment：** AudioSegment实例


## ResampleAugmentor

随机重采样的音频数据增强器


> **def `__init__`(self, prob=0.0, new_sample_rate=(8000, 16000, 24000)):**

**参数：**

 - **prob（float）：** 数据增强概率
 - **new_sample_rate（list）：** 新采样率列表

> **def `__call__`(self, audio_segment: AudioSegment) -> AudioSegment:**

**参数：**

 - **audio_segment：** AudioSegment实例


## NoisePerturbAugmentor

随机噪声扰动的音频数据增强器


> **def `__init__`(self, noise_dir='', prob=0.0, min_snr_dB=10, max_snr_dB=50):**

**参数：**

 - **noise_dir（str）：** 噪声文件夹路径，该文件夹下是噪声音频文件
 - **prob（float）：** 数据增强概率
 - **min_snr_dB（int）：** 最小信噪比
 - **max_snr_dB（int）：** 最大信噪比

> **def `__call__`(self, audio_segment: AudioSegment) -> AudioSegment:**

**参数：**

 - **audio_segment：** AudioSegment实例


## ReverbPerturbAugmentor

随机混响的音频数据增强器

> **def `__init__`(self, reverb_dir='', prob=0.0):**

**参数：**

 - **reverb_dir（str）：** 混响文件夹路径，该文件夹下是噪声音频文件
 - **prob（float）：** 数据增强概率

> **def `__call__`(self, audio_segment: AudioSegment) -> AudioSegment:**

**参数：**

 - **audio_segment：** AudioSegment实例


## SpecAugmentor

频域掩蔽和时域掩蔽的音频特征数据增强器<br/>
论文：https://arxiv.org/abs/1904.08779<br/>
论文：https://arxiv.org/abs/1912.05533

> **def `__init__`(self, prob=0.0,
                   freq_mask_ratio=0.15,
                   n_freq_masks=2,
                   time_mask_ratio=0.05,
                   n_time_masks=2,
                   inplace=True,
                   max_time_warp=5,
                   replace_with_zero=False):**

**参数：**

 - **prob（float）：** 数据增强概率
 - **freq_mask_ratio（float）：** 频域掩蔽的比例
 - **n_freq_masks（int）：** 频域掩蔽次数
 - **time_mask_ratio（float）：** 时间掩蔽的比例
 - **n_time_masks（int）：** 时间掩蔽次数
 - **inplace（bool）：** 用结果覆盖
 - **max_time_warp（bool）：** 最大时间扭曲
 - **replace_with_zero（bool）：** 是否使用0作为掩码，否则使用平均值

> **def `__call__`(self, x) -> np.ndarray:**

**参数：**

 - **x：** 音频特征，维度(time, freq)


## SpecSubAugmentor

从原始音频中随机替换部分帧，以模拟语音的时移。<br/>
论文：https://arxiv.org/abs/2106.05642

> **def `__init__`(self, prob=0.0, max_time=20, num_time_sub=3):**

**参数：**

 - **prob（float）：** 数据增强概率
 - **max_time（int）：** 时间替换的最大宽度
 - **num_time_sub（int）：** 时间替换的的次数

> **def `__call__`(self, x) -> np.ndarray:**

**参数：**

 - **x：** 音频特征，维度(time, freq)

<br/>
