# Copyright (c) 2024, Zhendong Peng (pzd17@tsinghua.org.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import vocos


class Vocos(vocos.Vocos):
    def __init__(self, name: str = "mel"):
        self.name = name
        assert name in ["encodec", "mel"]
        parent = vocos.Vocos.from_pretrained(f"charactr/vocos-{name}-24khz")
        super().__init__(parent.feature_extractor, parent.backbone, parent.head)
        if name == "encodec":
            self.feature_extractor.encodec.eval()
            self.bandwidths = self.feature_extractor.bandwidths
        self.feature_dim = self.backbone.input_channels
        # encodec: T => T * upsample_rate
        # mel: T => (T - 1) * upsample_rate
        self.upsample_rate = self.head.istft.hop_length

    def get_encodec_codes(self, audio: torch.Tensor, bandwidth_id: int = -1):
        assert self.name == "encodec"
        bandwidth = self.bandwidths[bandwidth_id]
        self.feature_extractor.encodec.set_target_bandwidth(bandwidth)
        return self.feature_extractor.get_encodec_codes(audio)

    def extra_features(self, audio: torch.Tensor, bandwidth_id: int = -1):
        if self.name == "encodec":
            codes = self.get_encodec_codes(audio, bandwidth_id)
            return self.codes_to_features(codes)
        return self.feature_extractor(audio)

    def decode(self, features: torch.Tensor, bandwidth_id: int = -1):
        if self.name == "encodec":
            if bandwidth_id < 0:
                bandwidth_id += len(self.bandwidths)
            assert 0 <= bandwidth_id < len(self.bandwidths)
            bandwidth_id = torch.tensor([bandwidth_id])
            return super().decode(features, bandwidth_id=bandwidth_id)
        return super().decode(features)

    def decode_codes(self, codes: torch.Tensor, bandwidth_id: int = -1):
        assert self.name == "encodec"
        features = self.codes_to_features(codes)
        return self.decode(features, bandwidth_id)


class StreamingVocos(Vocos):
    def __init__(
        self,
        name: str = "mel",
        bandwidth_id: int = -1,
        chunk_size_ms: int = 300,
        padding_ms: int = 320,
    ):
        super().__init__(name)
        self.bandwidth_id = bandwidth_id
        self.chunk_size = int(chunk_size_ms / 1000 * 24000 / self.upsample_rate)
        self.padding = int(padding_ms / 1000 * 24000 / self.upsample_rate)
        self.caches_len = self.chunk_size + 2 * self.padding

        self.cur_idx = 0
        self.caches = torch.zeros((1, self.feature_dim, self.caches_len))

    def reset(self):
        self.cur_idx = 0
        self.caches = torch.zeros((1, self.feature_dim, self.caches_len))

    def get_size(self):
        """
        Method to get the length of unprocessed codes or features.
        """
        effective_size = self.cur_idx + 1 - self.padding
        if effective_size <= 0:
            return 0
        return effective_size % self.chunk_size or self.chunk_size

    def streaming_decode(self, features: torch.Tensor, is_last: bool = False):
        """
        Method to streaming decode audio waveform from already calculated features.
        The features is passed through the backbone and the head to reconstruct the audio output.

        Args:
            features (Tensor): The input tensor of features of shape (B, C, L), where B is the batch size,
                               C denotes the feature dimension, and L is the sequence length.
            is_last (bool): Whether the input features is the last frame.

        Returns:
            Tensor: The output tensor representing the reconstructed audio waveform of shape (B, T).
        """
        num_features = features.shape[2]
        for idx, feature in enumerate(torch.unbind(features, dim=2)):
            self.cur_idx += 1
            self.caches = torch.roll(self.caches, shifts=-1, dims=2)
            self.caches[:, :, -1] = feature
            is_last = is_last and idx == num_features - 1
            cur_size = self.get_size()
            if cur_size != self.chunk_size and not is_last:
                continue
            audio = self.decode(self.caches, self.bandwidth_id)
            audio = audio[:, self.padding * self.upsample_rate :]
            if cur_size != self.chunk_size:
                audio = audio[:, (self.chunk_size - cur_size) * self.upsample_rate :]
            if not is_last:
                audio = audio[:, : self.chunk_size * self.upsample_rate]
            else:
                self.reset()
            yield audio

    def streaming_decode_codes(self, codes: torch.Tensor, is_last: bool = False):
        assert self.name == "encodec"
        features = self.codes_to_features(codes)
        for audio in self.streaming_decode(features, is_last=is_last):
            yield audio
