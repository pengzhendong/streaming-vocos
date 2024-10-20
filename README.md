# streaming-vocos

Streaming Vocos is a wrapper of [Vocos](https://github.com/gemelo-ai/vocos). It supports streaming reconstruction of audio from mel-spectrogram or EnCodec tokens.

## Usage

- From mel-spectrogram

``` python
from streaming_vocos import StreamingVocos

audios = []
vocos = StreamingVocos()
features = vocos.feature_extractor(audio)

for feature in torch.unbind(features, dim=2):
    audios += vocos.streaming_decode(feature[:, :, None])
audios.append(vocos.decode_caches())
audios = torch.cat(audios, dim=1)
```

- From EnCodec tokens

``` python
from streaming_vocos import StreamingVocos

audios = []
vocos = StreamingVocos()
codes = vocos.get_encodec_codes(audio)

for code in torch.unbind(codes, dim=2):
    audios += vocos.streaming_decode_codes(code[:, :, None])
audios.append(vocos.decode_caches())
audios = torch.cat(audios, dim=1)
```
