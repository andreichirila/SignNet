# Model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from Config import ModelConfig


# ============================================
# POSITIONAL ENCODING
# ============================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] tensor
        Returns:
            [B, T, D] tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================
# GRAPH CONVOLUTIONAL NETWORK (GCN)
# ============================================

class GraphConvolution(nn.Module):
    """Single Graph Convolution Layer."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, N, C_in] node features
            adj: [N, N] adjacency matrix
        Returns:
            [B, T, N, C_out] output features
        """
        # x: [B, T, N, C_in]
        # weight: [C_in, C_out]
        # adj: [N, N]

        support = torch.matmul(x, self.weight)  # [B, T, N, C_out]
        output = torch.matmul(adj, support)  # [N, N] @ [B, T, N, C_out] -> [B, T, N, C_out]

        if self.bias is not None:
            output = output + self.bias

        return output


class SpatialGCN(nn.Module):
    """
    Spatial Graph Convolutional Network for landmark features.
    Processes each frame independently to extract spatial features.
    """

    def __init__(
            self,
            num_landmarks: int,
            input_dim: int,
            hidden_dims: list,
            output_dim: int,
            dropout: float = 0.1
    ):
        super().__init__()

        self.num_landmarks = num_landmarks
        self.input_dim = input_dim

        # Build adjacency matrix
        self.register_buffer('adj', self._build_adjacency(num_landmarks))

        # GCN layers
        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(GraphConvolution(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.gcn_layers = nn.ModuleList(layers)

        # Final projection
        self.fc = nn.Linear(hidden_dims[-1] * num_landmarks, output_dim)
        self.dropout = nn.Dropout(dropout)

    def _build_adjacency(self, num_landmarks: int) -> torch.Tensor:
        """
        Build adjacency matrix for landmarks.
        Simple version: fully connected within each body part group.
        """
        adj = torch.zeros(num_landmarks, num_landmarks)

        # Hand connections (0-20 left, 21-41 right)
        # Simplified: connect each landmark to its neighbors
        for hand_start in [0, 21]:
            for i in range(21):
                idx = hand_start + i
                if idx < num_landmarks:
                    # Self-connection
                    adj[idx, idx] = 1.0
                    # Connect to neighbors in same hand
                    if i > 0:
                        adj[idx, idx - 1] = 1.0
                        adj[idx - 1, idx] = 1.0

        # Pose connections (42-74)
        for i in range(42, min(75, num_landmarks)):
            adj[i, i] = 1.0
            if i > 42:
                adj[i, i - 1] = 1.0
                adj[i - 1, i] = 1.0

        # Face: simplified - just self-connections for efficiency
        for i in range(75, num_landmarks):
            adj[i, i] = 1.0

        # Normalize adjacency
        degree = adj.sum(dim=1, keepdim=True)
        degree = torch.where(degree > 0, degree, torch.ones_like(degree))
        adj = adj / degree

        return adj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, N, C] landmark features
        Returns:
            [B, T, D] spatial features
        """
        B, T, N, C = x.shape

        # Process through GCN layers
        out = x
        for i, layer in enumerate(self.gcn_layers):
            if isinstance(layer, GraphConvolution):
                out = layer(out, self.adj)
            elif isinstance(layer, nn.BatchNorm1d):
                # Reshape for BatchNorm: [B*T*N, C]
                out_shape = out.shape
                out = out.reshape(-1, out_shape[-1])
                out = layer(out)
                out = out.reshape(out_shape)
            else:
                out = layer(out)

        # Flatten landmarks and project
        out = out.reshape(B, T, -1)  # [B, T, N*hidden]
        out = self.fc(out)  # [B, T, output_dim]
        out = self.dropout(out)

        return out


# ============================================
# MULTI-STREAM SPATIAL ENCODER
# ============================================

class MultiStreamSpatialEncoder(nn.Module):
    """
    Multi-stream encoder for different landmark groups.
    Separate GCNs for: left hand, right hand, pose+face
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        # Landmark group sizes
        self.left_hand_size = 21
        self.right_hand_size = 21
        self.pose_face_size = 501  # 33 pose + 468 face

        # Stream output dimension (each stream outputs this)
        stream_output_dim = config.d_model // 3  # Divide equally among streams

        # Left hand GCN
        self.left_hand_gcn = SpatialGCN(
            num_landmarks=self.left_hand_size,
            input_dim=config.gcn_input_dim,
            hidden_dims=config.gcn_hidden_dims,
            output_dim=stream_output_dim,
            dropout=config.gcn_dropout
        )

        # Right hand GCN
        self.right_hand_gcn = SpatialGCN(
            num_landmarks=self.right_hand_size,
            input_dim=config.gcn_input_dim,
            hidden_dims=config.gcn_hidden_dims,
            output_dim=stream_output_dim,
            dropout=config.gcn_dropout
        )

        # Pose + Face GCN (simplified - use linear instead for efficiency)
        self.pose_face_encoder = nn.Sequential(
            nn.Linear(self.pose_face_size * config.gcn_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(config.gcn_dropout),
            nn.Linear(512, stream_output_dim),
            nn.Dropout(config.gcn_dropout)
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(stream_output_dim * 3, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.ReLU(),
            nn.Dropout(config.gcn_dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, 543, 2] full landmark tensor
        Returns:
            [B, T, d_model] fused spatial features
        """
        B, T, N, C = x.shape

        # Split into streams
        left_hand = x[:, :, 0:21, :]  # [B, T, 21, 2]
        right_hand = x[:, :, 21:42, :]  # [B, T, 21, 2]
        pose_face = x[:, :, 42:, :]  # [B, T, 501, 2]

        # Process each stream
        left_features = self.left_hand_gcn(left_hand)  # [B, T, stream_dim]
        right_features = self.right_hand_gcn(right_hand)  # [B, T, stream_dim]

        # Pose+face: flatten and process
        pose_face_flat = pose_face.reshape(B, T, -1)  # [B, T, 501*2]
        pose_face_features = self.pose_face_encoder(pose_face_flat)  # [B, T, stream_dim]

        # Concatenate and fuse
        combined = torch.cat([left_features, right_features, pose_face_features], dim=-1)
        fused = self.fusion(combined)  # [B, T, d_model]

        return fused


# ============================================
# TRANSFORMER ENCODER
# ============================================

class TemporalTransformerEncoder(nn.Module):
    """Transformer encoder for temporal modeling."""

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=config.d_model,
            max_len=config.max_seq_length,
            dropout=config.transformer_dropout
        )

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.transformer_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for better training stability
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
            enable_nested_tensor=False
        )

        # Final layer norm
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(
            self,
            x: torch.Tensor,
            src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model] spatial features
            src_key_padding_mask: [B, T] True for padding positions
        Returns:
            [B, T, d_model] temporal features
        """
        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Final norm
        x = self.layer_norm(x)

        return x


# ============================================
# CTC DECODER
# ============================================

class CTCDecoder(nn.Module):
    """CTC decoder for sequence-to-sequence prediction."""

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.fc = nn.Linear(config.d_model, config.num_classes + 1)  # +1 for blank
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model] temporal features
        Returns:
            [B, T, num_classes+1] log probabilities
        """
        logits = self.fc(x)  # [B, T, num_classes+1]
        log_probs = self.log_softmax(logits)
        return log_probs


# ============================================
# FULL MODEL
# ============================================

class SignLanguageTransformer(nn.Module):
    """
    Complete Sign Language Recognition Model.
    Multi-stream GCN + Transformer + CTC decoder.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        # Spatial encoder (Multi-stream GCN)
        self.spatial_encoder = MultiStreamSpatialEncoder(config)

        # Temporal encoder (Transformer)
        self.temporal_encoder = TemporalTransformerEncoder(config)

        # CTC decoder
        self.ctc_decoder = CTCDecoder(config)

        # Initialize weights
        self._init_weights()

        # Print model info
        self._print_model_info()

    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def _print_model_info(self):
        """Print model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\n{'=' * 60}")
        print(f"MODEL: SignLanguageTransformer")
        print(f"{'=' * 60}")
        print(f"Total parameters:     {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size:           {total_params * 4 / 1024 / 1024:.1f} MB")
        print(f"{'=' * 60}")

    def create_padding_mask(
            self,
            lengths: torch.Tensor,
            max_len: int
    ) -> torch.Tensor:
        """
        Create padding mask for Transformer.

        Args:
            lengths: [B] actual lengths
            max_len: maximum sequence length
        Returns:
            [B, max_len] True for padding positions
        """
        batch_size = lengths.size(0)
        mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len)
        mask = mask >= lengths.unsqueeze(1)
        return mask

    def forward(
            self,
            landmarks: torch.Tensor,
            landmarks_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            landmarks: [B, T, 543, 2] landmark features
            landmarks_lengths: [B] actual sequence lengths

        Returns:
            log_probs: [B, T, num_classes+1] log probabilities for CTC
            output_lengths: [B] output sequence lengths
        """
        B, T, N, C = landmarks.shape

        # Create padding mask
        padding_mask = self.create_padding_mask(landmarks_lengths, T)

        # Spatial encoding (per-frame)
        spatial_features = self.spatial_encoder(landmarks)  # [B, T, d_model]

        # Temporal encoding (Transformer)
        temporal_features = self.temporal_encoder(
            spatial_features,
            src_key_padding_mask=padding_mask
        )  # [B, T, d_model]

        # CTC decoding
        log_probs = self.ctc_decoder(temporal_features)  # [B, T, vocab_size]

        # Output lengths are same as input for CTC
        output_lengths = landmarks_lengths

        return log_probs, output_lengths

    def decode_greedy(self, log_probs: torch.Tensor) -> list:
        """
        Greedy decoding for inference.

        Args:
            log_probs: [B, T, vocab_size] log probabilities

        Returns:
            List of decoded sequences (indices)
        """
        # Get most likely tokens
        predictions = log_probs.argmax(dim=-1)  # [B, T]

        decoded = []
        for seq in predictions:
            # Remove consecutive duplicates and blanks
            decoded_seq = []
            prev_token = -1

            for token in seq.tolist():
                if token != prev_token and token != 1:  # 1 is blank
                    decoded_seq.append(token)
                prev_token = token

            decoded.append(decoded_seq)

        return decoded


# ============================================
# TEST FUNCTION
# ============================================

def test_model():
    """Test model forward pass."""
    from Config import get_config

    print("\n" + "=" * 80)
    print("TESTING MODEL.PY")
    print("=" * 80)

    # Get config
    config = get_config(top_k=50, use_augmentation=False)

    # Create model
    print("\n🧠 Creating model...")
    model = SignLanguageTransformer(config.model)
    model = model.to(config.model.device)

    # Create dummy input
    print("\n📦 Creating dummy batch...")
    batch_size = 4
    seq_len = 100

    landmarks = torch.randn(batch_size, seq_len, 543, 2).to(config.model.device)
    lengths = torch.tensor([100, 80, 60, 50]).to(config.model.device)

    print(f"   Input shape: {landmarks.shape}")
    print(f"   Lengths: {lengths}")

    # Forward pass
    print("\n🔄 Running forward pass...")
    model.eval()
    with torch.no_grad():
        log_probs, output_lengths = model(landmarks, lengths)

    print(f"\n✅ Forward pass successful!")
    print(f"   Output shape: {log_probs.shape}")
    print(f"   Expected: [4, 100, {config.model.num_classes + 1}]")
    print(f"   Output lengths: {output_lengths}")

    # Test greedy decoding
    print("\n🔍 Testing greedy decoding...")
    decoded = model.decode_greedy(log_probs)
    print(f"   Decoded sequences: {len(decoded)}")
    for i, seq in enumerate(decoded[:3]):
        print(f"   Sample {i}: {len(seq)} tokens")

    # Memory usage
    if config.model.device == 'cuda':
        print(f"\n💾 GPU Memory:")
        print(f"   Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.1f} MB")
        print(f"   Cached: {torch.cuda.memory_reserved() / 1024 ** 2:.1f} MB")

    print("\n✅ Model.py test PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    test_model()