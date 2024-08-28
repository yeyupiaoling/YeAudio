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

<br/>

> **def `__init__`(self, samples, sample_rate):**

创建单通道音频片段实例

**参数：**

 - **samples（ndarray.float32）：** 频数据，维度为[num_samples x num_channels]
 - **sample_rate（int）：** 音频的采样率

<br/>

> **def `__eq__`(self, other):**

返回两个对象是否相等

**参数：**

 - **other（AudioSegment）：** 比较的另一个音频片段实例

<br/>

> **def `__ne__`(self, other):**

返回两个实例是否不相等

**参数：**

 - **other（AudioSegment）：** 比较的另一个音频片段实例

<br/>

> **def `__str__`(self):**

返回该音频的信息

<br/>

> **def from_file(cls, file):**

从音频文件创建音频段，支持wav、mp3、mp4等多种音频格式

**参数：**

 - **file（str|BufferedReader）：** 件路径，或者文件对象

**返回：**

 - `AudioSegment`：音频片段实例

<br/>

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

<br/>

> **def from_bytes(cls, data):**

从wav格式的音频字节创建音频段

**参数：**

 - **data（bytes）：** 包含音频样本的字节

**返回：**

 - `AudioSegment`：音频片段实例

<br/>

> **def from_pcm_bytes(cls, data, channels=1, samp_width=2, sample_rate=16000):**

从包含无格式PCM音频的字节创建音频

**参数：**

 - **data（bytes）：** 包含音频样本的字节
 - **channels（int）：** 音频的通道数
 - **samp_width（int）：** 频采样的宽度，如np.int16为2
 - **sample_rate（int）：** 音频样本采样率

**返回：**

 - `AudioSegment`：音频片段实例

<br/>

> **def from_ndarray(cls, data, sample_rate=16000):**

从numpy.ndarray创建音频段

**参数：**

 - **data（bytes）：** numpy.ndarray类型的音频数据
 - **sample_rate（int）：** 音频样本采样率

**返回：**

 - `AudioSegment`：音频片段实例

<br/>

> **def concatenate(cls, \*segments):**

将任意数量的音频片段连接在一起

**参数：**

 - **segments（AudioSegment）：** 输入音频片段被连接

**返回：**

 - `AudioSegment`：音频片段实例

**异常：**

 - `ValueError`：如果音频实例列表为空或者采样率不一致，则会抛出ValueError异常
 - `TypeError`：如果输入的片段类型不一致，则会抛出TypeError异常










