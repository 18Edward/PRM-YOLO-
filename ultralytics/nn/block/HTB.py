# htb.py - Histogram Transformer Block for YOLOv11
# Compatible with PyTorch >= 1.10, supports AMP training

import torch
import torch.nn as nn
import torch.nn.functional as F


# LayerNorm for 2D feature maps (B, C, H, W)
class LayerNorm(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim, affine=bias)  # Equivalent to LayerNorm(C) on 2D

    def forward(self, x):
        return self.norm(x)


# Dual-scale Gated Feed-Forward Network (DGFF)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.5, bias=False):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv_5 = nn.Conv2d(
            hidden_features // 4, hidden_features // 4,
            kernel_size=5, padding=2,
            groups=hidden_features // 4, bias=bias
        )
        self.dwconv_dilated = nn.Conv2d(
            hidden_features // 4, hidden_features // 4,
            kernel_size=3, padding=2,
            groups=hidden_features // 4, bias=bias, dilation=2
        )
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.pixel_unshuffle = nn.PixelUnshuffle(2)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)  # (B, 2C', H, W)
        x = self.pixel_shuffle(x)  # (B, C'/2, 2H, 2W)
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.dwconv_5(x1)
        x2 = self.dwconv_dilated(x2)
        x = F.mish(x2) * x1  # Gated fusion
        x = self.pixel_unshuffle(x)  # Back to (B, C', H, W)
        x = self.project_out(x)
        return x


# Dynamic-range Histogram Self-Attention (DHSA)
class Attention_histogram(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False, ifBox=True):
        super().__init__()
        self.num_heads = num_heads
        self.factor = num_heads
        self.ifBox = ifBox
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 5, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 5, dim * 5, kernel_size=3,
            padding=1, groups=dim * 5, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def pad_to_factor(self, x, factor):
        """Pad H or W to be divisible by factor"""
        b, c, h, w = x.shape
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [0, pad_w, 0, pad_h])
        return x, (h, w)

    def unpad(self, x, orig_size):
        """Remove padding"""
        h, w = orig_size
        return x[:, :, :h, :w]

    def reshape_for_attn(self, tensor, box=True):
        """
        Reshape for grouped attention.
        box=True: group along H (split H into H//factor chunks, each of size factor)
        box=False: group along W
        """
        t, orig_hw = self.pad_to_factor(tensor, self.factor)
        b, c, h, w = t.shape
        c_head = c // self.num_heads

        t = t.view(b, self.num_heads, c_head, h, w)
        if box:
            assert h % self.factor == 0, f"H={h} not divisible by factor={self.factor}"
            t = t.view(b, self.num_heads, c_head, h // self.factor, self.factor, w)
            t = t.permute(0, 1, 3, 2, 4, 5).contiguous()  # (b, head, h//f, c_head, w, f)
            t = t.view(b, self.num_heads, c_head * self.factor, (h // self.factor) * w)
        else:
            assert w % self.factor == 0, f"W={w} not divisible by factor={self.factor}"
            t = t.view(b, self.num_heads, c_head, h, w // self.factor, self.factor)
            t = t.permute(0, 1, 5, 2, 3, 4).contiguous()  # (b, head, f, c_head, h, w//f)
            t = t.view(b, self.num_heads, c_head * self.factor, h * (w // self.factor))
        return t

    def forward(self, x):
        b, c, h, w = x.shape

        # === Step 1: Sort spatially (simulate histogram) on first half channels ===
        x_half = x[:, :c // 2]
        x_sorted_h, idx_h = torch.sort(x_half, dim=2)  # sort along H
        x_sorted_hw, idx_w = torch.sort(x_sorted_h, dim=3)  # sort along W
        x[:, :c // 2] = x_sorted_hw

        # === Step 2: Project to QKV ===
        qkv = self.qkv_dwconv(self.qkv(x))
        q1, k1, q2, k2, v = qkv.chunk(5, dim=1)  # each: (B, C, H, W)

        # === Step 3: Flatten and sort along channel using v's order ===
        v_flat = v.view(b, c, -1)  # (b, c, h*w)
        _, idx = torch.sort(v_flat, dim=-1)  # idx: (b, c, h*w)

        def gather_with_idx(tensor):
            flat = tensor.view(b, c, -1)
            return torch.gather(flat, dim=-1, index=idx)

        q1_flat = gather_with_idx(q1)
        k1_flat = gather_with_idx(k1)
        q2_flat = gather_with_idx(q2)
        k2_flat = gather_with_idx(k2)
        v_sorted = gather_with_idx(v)

        # === Step 4: Reshape for grouped attention ===
        q1 = self.reshape_for_attn(q1_flat.view(b, c, h, w), box=True)
        k1 = self.reshape_for_attn(k1_flat.view(b, c, h, w), box=True)
        v1 = self.reshape_for_attn(v_sorted.view(b, c, h, w), box=True)
        q2 = self.reshape_for_attn(q2_flat.view(b, c, h, w), box=False)
        k2 = self.reshape_for_attn(k2_flat.view(b, c, h, w), box=False)
        v2 = self.reshape_for_attn(v_sorted.view(b, c, h, w), box=False)

        # === Step 5: Attention (disable AMP for stability) ===
        with torch.cuda.amp.autocast(enabled=False):
            attn1 = (q1 @ k1.transpose(-2, -1)) * self.temperature
            attn1 = F.softmax(attn1.float(), dim=-1).type_as(attn1)
            out1 = attn1 @ v1

            attn2 = (q2 @ k2.transpose(-2, -1)) * self.temperature
            attn2 = F.softmax(attn2.float(), dim=-1).type_as(attn2)
            out2 = attn2 @ v2

        out = out1 * out2  # Dual-path fusion
        out = out.view(b, c, h, w)
        out = self.unpad(out, (h, w))  # Remove padding

        # === Step 6: Inverse sort to restore original spatial layout ===
        out_half = out[:, :c // 2]  # Only restore half
        out_flat = out_half.view(b, c // 2, -1)  # (b, c//2, h*w)
        idx_w_flat = idx_w.view(b, c // 2, -1)  # (b, c//2, h*w)
        out_unsorted_w = torch.gather(out_flat, dim=-1, index=idx_w_flat)  # Reverse W sort

        # Now reverse H sort: permute to make H inner dim
        out_hw = out_unsorted_w.view(b, c // 2, h, w)
        out_trans = out_hw.permute(0, 1, 3, 2).contiguous().view(b * (c // 2) * w, h)
        idx_h_trans = idx_h.permute(0, 1, 3, 2).contiguous().view(b * (c // 2) * w, h)
        out_trans = torch.gather(out_trans, dim=-1, index=idx_h_trans)
        out_restored = out_trans.view(b, c // 2, w, h).permute(0, 1, 3, 2).contiguous()

        out[:, :c // 2] = out_restored

        # === Final projection ===
        out = self.project_out(out)
        return out


# Histogram Transformer Block (HTB)
class HTB(nn.Module):
    def __init__(self, c1, num_heads=4, ffn_expansion_factor=2.5, bias=False):
        super().__init__()
        self.attn = Attention_histogram(c1, num_heads, bias, ifBox=True)
        self.ffn = FeedForward(c1, ffn_expansion_factor, bias)
        self.norm1 = LayerNorm(c1, bias)
        self.norm2 = LayerNorm(c1, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# =================== 测试代码 ===================
# if __name__ == '__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     x = torch.randn(2, 64, 32, 32).to(device)  # B C H W
#
#     model = HTB(c1=64, num_heads=4).to(device)
#     y = model(x)
#
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape: {y.shape}")
#     assert x.shape == y.shape, "❌ Output shape mismatch!"
#     print("✅ HTB forward pass successful!")