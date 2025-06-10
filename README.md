# streaming-vocos

Streaming Vocos is a wrapper of [Vocos](https://github.com/gemelo-ai/vocos). It supports streaming reconstruction of audio from mel-spectrogram or EnCodec tokens.

## Usage

```python
import torch
from audiolab import load_audio
from streaming_vocos import StreamingVocos
from wavesurfer import play

audio, rate = load_audio("data/test_24k.wav", dtype="float32")
audio = torch.from_numpy(audio)
```

- From mel-spectrogram

```python
vocos = StreamingVocos("mel")
features = vocos.feature_extractor(audio)

def audio_generator():
    for feature in torch.unbind(features, dim=2):
        for chunk in vocos.streaming_decode(feature[:, :, None]):
            yield chunk
    yield vocos.decode_caches()

play(audio_generator(), rate, verbose=True)
```

- From EnCodec tokens

```python
vocos = StreamingVocos("encodec")
codes = vocos.get_encodec_codes(audio)

def audio_generator():
    for code in torch.unbind(codes, dim=2):
        for chunk in vocos.streaming_decode_codes(code[:, :, None]):
            yield chunk
    yield vocos.decode_caches()

play(audio_generator(), rate, verbose=True)
```
