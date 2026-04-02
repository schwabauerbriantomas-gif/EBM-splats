"""
Omnimodal Encoders — project text, images, and audio to S^639.

Each encoder maps its modality to the shared 640D hypersphere,
enabling cross-modal retrieval and the PGLF pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
import math


class TextEncoder(nn.Module):
    """
    Text encoder: token IDs → 640D hypersphere.
    
    Uses a Transformer encoder on top of token embeddings.
    Supports GPT-2 tokenizer input (vocab=50257).
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        dim: int = 640,
        n_layers: int = 6,
        n_heads: int = 10,
        ff_dim: int = 2560,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_seq_len, dim)
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.layer_norm = nn.LayerNorm(dim)
        
        # Mean pooling + projection head
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.01)
    
    def forward(
        self,
        input_ids: torch.Tensor,        # [B, S]
        attention_mask: Optional[torch.Tensor] = None,  # [B, S]
    ) -> torch.Tensor:
        """Returns [B, D] embeddings on the unit hypersphere."""
        B, S = input_ids.shape
        device = input_ids.device
        
        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        x = self.dropout(self.token_embed(input_ids) + self.pos_embed(positions))
        
        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(S, device=device)
        
        # Padding mask
        if attention_mask is not None:
            pad_mask = (attention_mask == 0)
        else:
            pad_mask = None
        
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=pad_mask)
        x = self.layer_norm(x)
        
        # Mean pooling (excluding padding)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        
        x = self.proj(x)
        
        # Project to hypersphere
        return F.normalize(x, dim=-1)


class ImageEncoder(nn.Module):
    """
    Image encoder: CLIP ViT-B/32 features → 640D hypersphere.
    
    Loads a pre-trained CLIP model (frozen) and adds a projection head.
    Falls back to a simple CNN if CLIP is not available.
    """
    
    def __init__(
        self,
        clip_dim: int = 512,  # ViT-B/32 output dim
        dim: int = 640,
        freeze_clip: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.clip_dim = clip_dim
        self.clip_loaded = False
        
        # Try to load CLIP
        try:
            import clip
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device="cpu")
            if freeze_clip:
                for p in self.clip_model.parameters():
                    p.requires_grad = False
            self.clip_loaded = True
        except ImportError:
            pass
        
        # Projection head (always trainable)
        self.proj = nn.Sequential(
            nn.Linear(clip_dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, 3, 224, 224] normalized images
        Returns:
            [B, D] embeddings on hypersphere
        """
        if self.clip_loaded:
            with torch.no_grad():
                features = self.clip_model.encode_image(images).float()
        else:
            # Fallback: simple projection of flattened image
            B = images.shape[0]
            features = images.mean(dim=[2, 3])  # [B, 3] global avg pool
            if features.shape[-1] != self.clip_dim:
                # Pad or project to clip_dim
                pad = torch.zeros(B, self.clip_dim - features.shape[-1], device=features.device)
                features = torch.cat([features, pad], dim=-1)
        
        x = self.proj(features)
        return F.normalize(x, dim=-1)
    
    def encode_text_as_image_fallback(self, text_features: torch.Tensor) -> torch.Tensor:
        """If no images, use CLIP text features directly."""
        if self.clip_loaded:
            with torch.no_grad():
                x = self.proj(text_features.float())
                return F.normalize(x, dim=-1)
        raise RuntimeError("CLIP not loaded")


class AudioEncoder(nn.Module):
    """
    Audio encoder: Whisper features → 640D hypersphere.
    
    Uses pre-trained Whisper base encoder (frozen) + projection.
    Falls back to a simple conv encoder if Whisper is not available.
    """
    
    def __init__(
        self,
        whisper_dim: int = 768,  # Whisper base encoder dim
        dim: int = 640,
        freeze_whisper: bool = True,
        n_mels: int = 80,
    ):
        super().__init__()
        self.dim = dim
        self.whisper_dim = whisper_dim
        self.whisper_loaded = False
        self.n_mels = n_mels
        
        # Try to load Whisper
        try:
            import whisper
            self.whisper_model = whisper.load_model("base")
            if freeze_whisper:
                for p in self.whisper_model.parameters():
                    p.requires_grad = False
                self.whisper_model.eval()
            self.whisper_loaded = True
        except ImportError:
            pass
        
        # Fallback conv encoder
        if not self.whisper_loaded:
            self.conv_encoder = nn.Sequential(
                nn.Conv1d(n_mels, 256, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(256, 512, kernel_size=3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
            )
            whisper_dim = 512
        
        # Projection head
        self.proj = nn.Sequential(
            nn.Linear(whisper_dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
    
    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_spectrogram: [B, n_mels, T] mel features
        Returns:
            [B, D] embeddings on hypersphere
        """
        if self.whisper_loaded:
            with torch.no_grad():
                # Whisper encoder expects [B, n_mels, 3000]
                # Pad or truncate to expected length
                if mel_spectrogram.shape[-1] < 3000:
                    mel_spectrogram = F.pad(mel_spectrogram, (0, 3000 - mel_spectrogram.shape[-1]))
                else:
                    mel_spectrogram = mel_spectrogram[:, :, :3000]
                
                # Get encoder features
                features = self.whisper_model.encoder(mel_spectrogram)
                # Average pool over time
                features = features.mean(dim=1)  # [B, whisper_dim]
        else:
            features = self.conv_encoder(mel_spectrogram).squeeze(-1)  # [B, 512]
        
        x = self.proj(features)
        return F.normalize(x, dim=-1)


class OmnimodalEncoder(nn.Module):
    """
    Unified interface for all modalities.
    
    Routes input to the correct encoder and returns normalized embeddings
    on the shared 640D hypersphere.
    """
    
    MODALITY_TEXT = 0
    MODALITY_IMAGE = 1
    MODALITY_AUDIO = 2
    
    def __init__(self, dim: int = 640, freeze_pretrained: bool = True):
        super().__init__()
        self.dim = dim
        
        self.text_encoder = TextEncoder(dim=dim)
        self.image_encoder = ImageEncoder(freeze_clip=freeze_pretrained)
        self.audio_encoder = AudioEncoder(freeze_whisper=freeze_pretrained)
    
    def encode_text(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: (embeddings [B, D], modality [B])
        """
        emb = self.text_encoder(input_ids, attention_mask)
        modality = torch.full((emb.shape[0],), self.MODALITY_TEXT, device=emb.device, dtype=torch.long)
        return emb, modality
    
    def encode_image(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: (embeddings [B, D], modality [B])
        """
        emb = self.image_encoder(images)
        modality = torch.full((emb.shape[0],), self.MODALITY_IMAGE, device=emb.device, dtype=torch.long)
        return emb, modality
    
    def encode_audio(self, mel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: (embeddings [B, D], modality [B])
        """
        emb = self.audio_encoder(mel)
        modality = torch.full((emb.shape[0],), self.MODALITY_AUDIO, device=emb.device, dtype=torch.long)
        return emb, modality
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        mel_spectrogram: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode any combination of modalities.
        
        Returns: (all_embeddings [N, D], all_modalities [N])
        """
        embeddings = []
        modalities = []
        
        if input_ids is not None:
            emb, mod = self.encode_text(input_ids, attention_mask)
            embeddings.append(emb)
            modalities.append(mod)
        
        if images is not None:
            emb, mod = self.encode_image(images)
            embeddings.append(emb)
            modalities.append(mod)
        
        if mel_spectrogram is not None:
            emb, mod = self.encode_audio(mel_spectrogram)
            embeddings.append(emb)
            modalities.append(mod)
        
        if not embeddings:
            raise ValueError("At least one modality must be provided")
        
        return torch.cat(embeddings, dim=0), torch.cat(modalities, dim=0)


# Fix missing import
from typing import Tuple
