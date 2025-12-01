"""
🧠 Multi-Stream Sign Language Transformer

Architecture based on SL-GCN multi-stream concept:
- Separate encoder branches for each stream
- Learnable fusion mechanism
- Shared transformer decoder

Author: Andrei Chirila, Roman Schläpfer
Date: 2025-12-01
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class MultiStreamModelConfig:
    """Configuration for Multi-Stream Transformer."""

    # Input
    num_landmarks: int = 543
    landmark_dim: int = 2
    num_bones: int = 70

    # Streams
    use_joint: bool = True
    use_bone: bool = True
    use_joint_motion: bool = True
    use_bone_motion: bool = True

    # Stream encoder
    stream_hidden_dim: int = 256
    stream_num_layers: int = 2

    # Fusion
    fusion_type: str = 'attention'  # 'concat', 'attention', 'gated', 'weighted'

    # Transformer
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1

    # Output
    num_classes: int = 203
    max_seq_length: int = 214
    device: str = 'cuda'

    @property
    def num_streams(self) -> int:
        return sum([self.use_joint, self.use_bone,
                    self.use_joint_motion, self.use_bone_motion])


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class StreamEncoder(nn.Module):
    """Encoder for a single stream using 1D convolutions."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()

        layers = []
        layers.append(nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=kernel_size // 2))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 2):
            layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        if num_layers > 1:
            layers.append(nn.Conv1d(hidden_dim, output_dim, kernel_size, padding=kernel_size // 2))
            layers.append(nn.BatchNorm1d(output_dim))
            layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> Conv expects (B, D, T)
        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = x.transpose(1, 2)
        return x


class MultiStreamFusion(nn.Module):
    """Fuse multiple stream encodings."""

    def __init__(self, num_streams: int, stream_dim: int, output_dim: int,
                 fusion_type: str = 'attention', dropout: float = 0.1):
        super().__init__()

        self.num_streams = num_streams
        self.fusion_type = fusion_type

        if fusion_type == 'concat':
            self.projection = nn.Linear(stream_dim * num_streams, output_dim)

        elif fusion_type == 'attention':
            self.query = nn.Linear(stream_dim, stream_dim // 4)
            self.key = nn.Linear(stream_dim, stream_dim // 4)
            self.value = nn.Linear(stream_dim, stream_dim)
            self.projection = nn.Linear(stream_dim, output_dim)

        elif fusion_type == 'gated':
            self.gates = nn.ModuleList([
                nn.Sequential(nn.Linear(stream_dim, stream_dim), nn.Sigmoid())
                for _ in range(num_streams)
            ])
            self.projection = nn.Linear(stream_dim, output_dim)

        elif fusion_type == 'weighted':
            self.stream_weights = nn.Parameter(torch.ones(num_streams) / num_streams)
            self.projection = nn.Linear(stream_dim, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, streams: List[torch.Tensor]) -> torch.Tensor:
        if self.fusion_type == 'concat':
            fused = torch.cat(streams, dim=-1)
            fused = self.projection(fused)

        elif self.fusion_type == 'attention':
            stacked = torch.stack(streams, dim=2)  # (B, T, S, D)
            B, T, S, D = stacked.shape

            q = self.query(stacked.mean(dim=2, keepdim=True))
            k = self.key(stacked)
            v = self.value(stacked)

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D // 4)
            weights = F.softmax(scores, dim=-1)
            fused = torch.matmul(weights, v).squeeze(2)
            fused = self.projection(fused)

        elif self.fusion_type == 'gated':
            gated = [stream * gate(stream) for stream, gate in zip(streams, self.gates)]
            fused = sum(gated) / self.num_streams
            fused = self.projection(fused)

        elif self.fusion_type == 'weighted':
            weights = F.softmax(self.stream_weights, dim=0)
            stacked = torch.stack(streams, dim=2)
            fused = (stacked * weights.view(1, 1, -1, 1)).sum(dim=2)
            fused = self.projection(fused)

        return self.layer_norm(self.dropout(fused))


class MultiStreamSignLanguageTransformer(nn.Module):
    """Multi-Stream Transformer for Continuous Sign Language Recognition."""

    def __init__(self, config: MultiStreamModelConfig):
        super().__init__()
        self.config = config

        joint_dim = config.num_landmarks * config.landmark_dim
        bone_dim = config.num_bones * config.landmark_dim

        # Stream encoders
        self.stream_encoders = nn.ModuleDict()

        if config.use_joint:
            self.stream_encoders['joint'] = StreamEncoder(
                joint_dim, config.stream_hidden_dim, config.stream_hidden_dim,
                config.stream_num_layers, dropout=config.dropout)

        if config.use_bone:
            self.stream_encoders['bone'] = StreamEncoder(
                bone_dim, config.stream_hidden_dim, config.stream_hidden_dim,
                config.stream_num_layers, dropout=config.dropout)

        if config.use_joint_motion:
            self.stream_encoders['joint_motion'] = StreamEncoder(
                joint_dim, config.stream_hidden_dim, config.stream_hidden_dim,
                config.stream_num_layers, dropout=config.dropout)

        if config.use_bone_motion:
            self.stream_encoders['bone_motion'] = StreamEncoder(
                bone_dim, config.stream_hidden_dim, config.stream_hidden_dim,
                config.stream_num_layers, dropout=config.dropout)

        self.stream_names = list(self.stream_encoders.keys())

        # Fusion
        self.fusion = MultiStreamFusion(
            len(self.stream_names), config.stream_hidden_dim, config.d_model,
            config.fusion_type, config.dropout)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.d_model, config.max_seq_length, config.dropout)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.n_heads,
            dim_feedforward=config.d_ff, dropout=config.dropout,
            activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, config.n_layers)

        # Output
        self.output_proj = nn.Linear(config.d_model, config.num_classes)

        self._init_weights()
        self._log_architecture()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _log_architecture(self):
        params = sum(p.numel() for p in self.parameters())
        print(f"\n📊 MultiStreamTransformer:")
        print(f"   Streams: {self.stream_names}")
        print(f"   Fusion: {self.config.fusion_type}")
        print(f"   Parameters: {params:,}")

    def forward(self, streams: Dict[str, torch.Tensor], lengths: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            streams: Dict with 'joint', 'bone', 'joint_motion', 'bone_motion'
                     Each: (B, T, N, 2) or (B, T, D)
            lengths: (B,) sequence lengths
        Returns:
            log_probs: (B, T, num_classes)
            output_lengths: (B,)
        """
        device = lengths.device

        # Encode each stream
        encoded = []
        for name in self.stream_names:
            if name in streams:
                x = streams[name]
                if x.dim() == 4:  # (B, T, N, D) -> (B, T, N*D)
                    x = x.reshape(x.size(0), x.size(1), -1)
                encoded.append(self.stream_encoders[name](x))

        # Fuse
        fused = self.fusion(encoded)
        fused = self.pos_encoder(fused)

        # Mask
        max_len = fused.size(1)
        mask = torch.arange(max_len, device=device).unsqueeze(0) >= lengths.unsqueeze(1)

        # Transform
        out = self.transformer(fused, src_key_padding_mask=mask)
        logits = self.output_proj(out)
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs, lengths.clone()


if __name__ == "__main__":
    # Test
    config = MultiStreamModelConfig(d_model=256, n_layers=2)
    model = MultiStreamSignLanguageTransformer(config)

    B, T = 2, 100
    streams = {
        'joint': torch.randn(B, T, 543, 2),
        'bone': torch.randn(B, T, 70, 2),
        'joint_motion': torch.randn(B, T, 543, 2),
        'bone_motion': torch.randn(B, T, 70, 2),
    }
    lengths = torch.tensor([100, 80])

    log_probs, out_lens = model(streams, lengths)
    print(f"Output: {log_probs.shape}")