# src/model.py
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=512, pretrained: bool = False):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)
        # remove final fc
        modules = list(resnet.children())[:-1]
        self.backbone = nn.Sequential(*modules)  # output: (B, 512, 1, 1)
        self.fc = nn.Linear(512, embed_dim)

    def forward(self, x):
        # x: (B, C, H, W) or for sequence: (B, seq, C, H, W)
        if x.dim() == 5:
            B, S, C, H, W = x.shape
            x = x.view(B * S, C, H, W)
            f = self.backbone(x)  # (B*S, 512, 1, 1)
            f = f.view(B * S, -1)
            f = self.fc(f)
            f = f.view(B, S, -1)
            return f  # (B, S, embed_dim)
        else:
            f = self.backbone(x)
            f = f.view(x.size(0), -1)
            return self.fc(f)  # (B, embed_dim)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=300, hidden_dim=256, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: (B, seq_len, max_text_len) or single: (B, max_text_len)
        if x.dim() == 3:
            B, S, L = x.shape
            x = x.view(B * S, L)
            e = self.embedding(x)  # (B*S, L, emb_dim)
            _, (h, _) = self.lstm(e)
            h = h[-1]
            h = self.fc(h)  # (B*S, hidden_dim)
            h = h.view(B, S, -1)
            return h  # (B, S, hidden_dim)
        else:
            e = self.embedding(x)
            _, (h, _) = self.lstm(e)
            return self.fc(h[-1])


class CrossModalAttentionFusion(nn.Module):
    """
    Simple cross-modal fusion where text queries attend to image vectors.
    image_emb: (B, S, d)
    text_emb: (B, S, d_t)
    We project to same dim and do scaled dot-product attention per timestep then average.
    """

    def __init__(self, image_dim, text_dim, out_dim=512):
        super().__init__()
        self.q_proj = nn.Linear(text_dim, out_dim)  # queries from text
        self.k_proj = nn.Linear(image_dim, out_dim)  # keys from image
        self.v_proj = nn.Linear(image_dim, out_dim)
        self.out = nn.Linear(out_dim + text_dim + image_dim, out_dim)
        self.scale = out_dim ** 0.5

    def forward(self, image_emb, text_emb):
        # image_emb, text_emb: (B, S, d)
        Q = self.q_proj(text_emb)  # (B,S,out)
        K = self.k_proj(image_emb)
        V = self.v_proj(image_emb)
        # attention per sequence element
        # compute similarity across sequence positions (SxS)
        attn_scores = torch.matmul(Q, K.transpose(1, 2)) / self.scale  # (B, S, S)
        attn = torch.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn, V)  # (B, S, out)
        # combine with original embeddings (concat along last dim)
        combined = torch.cat([attn_out, text_emb, image_emb], dim=-1)  # (B,S, out+text+img)
        fused = self.out(combined)  # (B,S,out_dim)
        # optionally pool across time: here we return sequence of fused embeddings (B,S,out)
        return fused


class ConcatFusion(nn.Module):
    def __init__(self, image_dim, text_dim, out_dim=512):
        super().__init__()
        self.out = nn.Linear(image_dim + text_dim, out_dim)

    def forward(self, image_emb, text_emb):
        # both (B,S,d)
        c = torch.cat([image_emb, text_emb], dim=-1)
        return self.out(c)


class SequenceModel(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        # x: (B, S, dim)
        out, h = self.gru(x)
        # return last hidden state
        return h[-1]  # (B, hidden_dim)


class TextDecoder(nn.Module):
    def __init__(self, hidden_dim, vocab_size, max_len=20, emb_dim=256):
        super().__init__()
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, context, targets=None):
        # context: (B, hidden_dim). We'll use it as initial hidden state.
        B = context.size(0)
        h0 = context.unsqueeze(0)  # (1, B, hidden_dim)
        c0 = torch.zeros_like(h0)
        # if targets given (teacher forcing)
        if targets is not None:
            # targets: (B, L) token ids
            emb = self.embedding(targets)  # (B, L, emb_dim)
            out, _ = self.lstm(emb, (h0, c0))
            logits = self.fc(out)  # (B, L, V)
            return logits
        # otherwise greedy decode
        outputs = []
        input_tok = torch.full((B, 1), 2, dtype=torch.long, device=context.device)  # BOS idx=2
        for _ in range(self.max_len):
            emb = self.embedding(input_tok)  # (B,1,emb)
            out, (h0, c0) = self.lstm(emb, (h0, c0))
            logits = self.fc(out.squeeze(1))  # (B, V)
            next_tok = logits.argmax(dim=-1, keepdim=True)  # (B,1)
            outputs.append(next_tok)
            input_tok = next_tok
        outs = torch.cat(outputs, dim=1)  # (B, L)
        return outs


class ImageDecoder(nn.Module):
    """
    Very small decoder: maps context -> low-res image (3x64x64) via FC + ConvTranspose layers.
    """

    def __init__(self, hidden_dim=512, out_channels=3, ngf=64):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, ngf * 8 * 4 * 4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(ngf, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, context):
        # context: (B, hidden_dim)
        x = self.fc(context)
        B = x.size(0)
        x = x.view(B, -1, 4, 4)
        x = self.dec(x)
        # output range [-1,1] (use tanh). For visualization convert accordingly.
        return x


class StoryModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        seq_len=3,
        image_embed=512,
        text_hidden=256,
        fusion_type="cross",
        device=None,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.image_encoder = ImageEncoder(embed_dim=image_embed)
        self.text_encoder = TextEncoder(vocab_size=vocab_size, emb_dim=256, hidden_dim=text_hidden)
        if fusion_type == "cross":
            self.fusion = CrossModalAttentionFusion(image_dim=image_embed, text_dim=text_hidden, out_dim=512)
        else:
            self.fusion = ConcatFusion(image_dim=image_embed, text_dim=text_hidden, out_dim=512)
        self.seq_model = SequenceModel(input_dim=512, hidden_dim=512)
        self.text_decoder = TextDecoder(hidden_dim=512, vocab_size=vocab_size, max_len=20)
        self.image_decoder = ImageDecoder(hidden_dim=512)

    def forward(self, images, texts, target_texts=None):
        """
        images: (B, S, C, H, W)
        texts: (B, S, L)
        target_texts: (B, L) optional for teacher forcing
        """
        img_emb = self.image_encoder(images)  # (B,S,dim)
        txt_emb = self.text_encoder(texts)  # (B,S,dim_t)
        fused = self.fusion(img_emb, txt_emb)  # (B,S,out)
        context = self.seq_model(fused)  # (B, hidden_dim)
        # text logits if target_texts provided, else generate ids
        text_out = self.text_decoder(context, targets=target_texts)  # (B,L,V) or (B,L_ids)
        img_out = self.image_decoder(context)  # (B,3,H,W)
        return text_out, img_out
