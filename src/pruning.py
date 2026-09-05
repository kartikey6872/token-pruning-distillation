import torch


def compute_token_importance(attn_weights, cls_index=0):
    """
    Given attention weights from one layer (B, num_tokens, num_tokens),
    compute an importance score for each token based on how much
    attention the CLS token pays to it.
    """
    # attn_weights shape: (B, num_tokens, num_tokens)
    # row = query, col = key. We look at how much CLS (row 0) attends to each token.
    cls_attn = attn_weights[:, cls_index, :]  # (B, num_tokens) — attention FROM cls TO all tokens
    return cls_attn  # higher value = more important


def prune_tokens(x, attn_weights, keep_ratio=0.7, cls_index=0):
    """
    Drops the least important tokens, always keeping the CLS token.

    x: (B, num_tokens, embed_dim) — the current sequence
    attn_weights: (B, num_tokens, num_tokens) — attention from the layer just run
    keep_ratio: fraction of NON-CLS tokens to keep (e.g., 0.7 = keep 70%)

    Returns pruned x of shape (B, new_num_tokens, embed_dim)
    """
    B, num_tokens, embed_dim = x.shape
    importance = compute_token_importance(attn_weights, cls_index)  # (B, num_tokens)

    # We never want to prune the CLS token itself, so mask it out of ranking
    importance_no_cls = importance.clone()
    importance_no_cls[:, cls_index] = float("inf")  # ensures CLS always ranks as "most important"

    num_patch_tokens = num_tokens - 1  # excluding CLS
    num_keep = max(1, int(num_patch_tokens * keep_ratio))
    num_keep_total = num_keep + 1  # +1 for CLS token

    # Get indices of the top-scoring tokens (per batch item)
    topk_indices = importance_no_cls.topk(num_keep_total, dim=1).indices  # (B, num_keep_total)
    topk_indices, _ = torch.sort(topk_indices, dim=1)  # keep original order (CLS stays first)

    # Gather the kept tokens
    batch_indices = torch.arange(B).unsqueeze(1).expand(-1, num_keep_total)
    pruned_x = x[batch_indices, topk_indices]  # (B, num_keep_total, embed_dim)

    return pruned_x


if __name__ == "__main__":
    # sanity check with dummy data
    B, num_tokens, embed_dim = 2, 65, 128  # 64 patches + 1 CLS token
    dummy_x = torch.randn(B, num_tokens, embed_dim)
    dummy_attn = torch.softmax(torch.randn(B, num_tokens, num_tokens), dim=-1)

    pruned = prune_tokens(dummy_x, dummy_attn, keep_ratio=0.5)
    print("Original shape:", dummy_x.shape)
    print("Pruned shape:", pruned.shape)