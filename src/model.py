import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Splits image into patches and embeds them."""
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=128):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                  # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class TransformerBlock(nn.Module):
    """One transformer encoder block, returns output + attention weights."""
    def __init__(self, embed_dim=128, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(normed, normed, normed, need_weights=True, average_attn_weights=True)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


class SimpleViT(nn.Module):
    """Small Vision Transformer, sized for CPU/8GB RAM training."""
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 embed_dim=128, depth=6, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x, return_attn=False, prune_after_layer=None, keep_ratio=0.7):
        from pruning import prune_tokens  # local import to avoid circular import issues

        B = x.shape[0]
        x = self.patch_embed(x)                          # (B, num_patches, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)             # (B, num_patches+1, embed_dim)
        x = x + self.pos_embed
        x = self.dropout(x)

        attn_weights_list = []
        for i, block in enumerate(self.blocks):
            x, attn_weights = block(x)
            attn_weights_list.append(attn_weights)

            if prune_after_layer is not None and i == prune_after_layer:
                x = prune_tokens(x, attn_weights, keep_ratio=keep_ratio)

        x = self.norm(x)
        cls_output = x[:, 0]                              # use CLS token for classification
        logits = self.head(cls_output)

        if return_attn:
            return logits, attn_weights_list
        return logits


if __name__ == "__main__":
    # quick sanity check — run this file directly to test the model builds and runs
    model = SimpleViT()
    dummy_input = torch.randn(2, 3, 32, 32)  # batch of 2 images, 3 channels, 32x32

    # test without pruning
    output = model(dummy_input)
    print("Output shape (no pruning):", output.shape)

    # test with pruning
    output_pruned = model(dummy_input, prune_after_layer=2, keep_ratio=0.5)
    print("Output shape (with pruning after layer 2):", output_pruned.shape)

    print("Number of parameters:", sum(p.numel() for p in model.parameters()))