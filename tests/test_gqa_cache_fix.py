"""
Evaluation script for GroupedQueryAttention KV cache position fix.

Tests:
  1. Training mode (no cache) — causal mask is standard lower-triangular
  2. Single-token generation steps — matches full-sequence recompute
  3. Multi-token prefill + generation — matches full-sequence recompute
  4. Position tracking state — ptr_current_pos is correct after each step
  5. Causal mask correctness — no future-token leakage
"""

import torch
import sys

sys.path.insert(0, "src")

from BLLMC.components.config import GPT_Config
from BLLMC.components.attention.grouped_query import GroupedQueryAttention


def make_config(**kw):
    defaults = dict(
        emb_dim=64,
        n_heads=4,
        n_kv_heads=2,
        context_length=32,
        rope_base=10_000.0,
        drop_rate=0.0,
        vocab_size=100,
        n_layers=1,
    )
    defaults.update(kw)
    return GPT_Config(**defaults)


def test_training_mode_no_cache():
    """Without cache, output should be identical on repeated calls (no state leak)."""
    print("=" * 60)
    print("TEST 1: Training mode (no cache) — deterministic & no state leak")
    cfg = make_config()
    attn = GroupedQueryAttention(cfg).eval()

    x = torch.randn(1, 8, 64)
    out1 = attn(x, use_cache=False)
    out2 = attn(x, use_cache=False)

    assert torch.allclose(out1, out2, atol=1e-6), "Training mode outputs differ!"
    assert (
        attn.ptr_current_pos == 0
    ), f"ptr_current_pos should be 0, got {attn.ptr_current_pos}"
    assert attn.k_cache is None, "k_cache should be None after no-cache forward"
    print("  PASSED ✓")


def test_single_token_generation_matches_full():
    """
    Generate tokens one-at-a-time with cache. The output at each position
    should match a full-sequence recompute (no cache) on the same input.
    """
    print("=" * 60)
    print("TEST 2: Single-token generation vs full-sequence recompute")
    cfg = make_config()
    attn = GroupedQueryAttention(cfg).eval()

    seq_len = 6
    x_full = torch.randn(1, seq_len, 64)

    # Full recompute (no cache)
    with torch.no_grad():
        out_full = attn(x_full, use_cache=False)

    # Token-by-token with cache
    attn.reset_cache()
    cached_outputs = []
    with torch.no_grad():
        for i in range(seq_len):
            x_tok = x_full[:, i : i + 1, :]  # (1, 1, 64)
            out_tok = attn(x_tok, use_cache=True)
            cached_outputs.append(out_tok)

    out_cached = torch.cat(cached_outputs, dim=1)  # (1, seq_len, 64)

    max_diff = (out_full - out_cached).abs().max().item()
    print(f"  Max difference: {max_diff:.2e}")

    assert torch.allclose(
        out_full, out_cached, atol=1e-5
    ), f"Cached single-token output diverges from full recompute! Max diff: {max_diff:.2e}"
    assert (
        attn.ptr_current_pos == seq_len
    ), f"ptr_current_pos should be {seq_len}, got {attn.ptr_current_pos}"
    print("  PASSED ✓")


def test_multitok_prefill_plus_generation():
    """
    Prefill with a chunk of tokens, then generate one-at-a-time.
    Output should match full-sequence recompute.
    """
    print("=" * 60)
    print("TEST 3: Multi-token prefill (4 tokens) + single-token generation (2 tokens)")
    cfg = make_config()
    attn = GroupedQueryAttention(cfg).eval()

    prefill_len = 4
    gen_len = 2
    total_len = prefill_len + gen_len
    x_full = torch.randn(1, total_len, 64)

    # Full recompute
    with torch.no_grad():
        out_full = attn(x_full, use_cache=False)

    # Prefill + generate
    attn.reset_cache()
    with torch.no_grad():
        # Prefill
        x_pre = x_full[:, :prefill_len, :]
        out_pre = attn(x_pre, use_cache=True)

        # Generate one token at a time
        gen_outputs = [out_pre]
        for i in range(prefill_len, total_len):
            x_tok = x_full[:, i : i + 1, :]
            out_tok = attn(x_tok, use_cache=True)
            gen_outputs.append(out_tok)

    out_cached = torch.cat(gen_outputs, dim=1)

    max_diff = (out_full - out_cached).abs().max().item()
    print(f"  Max difference: {max_diff:.2e}")

    assert torch.allclose(
        out_full, out_cached, atol=1e-5
    ), f"Prefill+gen output diverges! Max diff: {max_diff:.2e}"
    assert (
        attn.ptr_current_pos == total_len
    ), f"ptr_current_pos should be {total_len}, got {attn.ptr_current_pos}"
    print("  PASSED ✓")


def test_causal_mask_no_future_leakage():
    """
    Verify that the causal mask correctly prevents attending to future tokens.
    Token at position i should produce the SAME output regardless of what
    tokens come after it.
    """
    print("=" * 60)
    print("TEST 4: Causal mask — no future-token leakage")
    cfg = make_config()
    attn = GroupedQueryAttention(cfg).eval()

    x_base = torch.randn(1, 5, 64)

    # Get output for position 2 with only 3 tokens present
    with torch.no_grad():
        out_short = attn(x_base[:, :3, :], use_cache=False)
    pos2_short = out_short[:, 2, :]

    # Get output for position 2 with all 5 tokens present
    with torch.no_grad():
        out_long = attn(x_base, use_cache=False)
    pos2_long = out_long[:, 2, :]

    max_diff = (pos2_short - pos2_long).abs().max().item()
    print(f"  Max difference at position 2: {max_diff:.2e}")

    assert torch.allclose(
        pos2_short, pos2_long, atol=1e-6
    ), f"Causal mask leaks future tokens! Max diff: {max_diff:.2e}"
    print("  PASSED ✓")


def test_ptr_position_tracking():
    """Verify ptr_current_pos is updated correctly across multiple steps."""
    print("=" * 60)
    print("TEST 5: Position pointer tracking")
    cfg = make_config()
    attn = GroupedQueryAttention(cfg).eval()
    x = torch.randn(1, 10, 64)

    assert attn.ptr_current_pos == 0, "Initial pos should be 0"

    # Prefill 4 tokens
    with torch.no_grad():
        attn(x[:, :4, :], use_cache=True)
    assert (
        attn.ptr_current_pos == 4
    ), f"After prefill(4): expected 4, got {attn.ptr_current_pos}"

    # Generate 1 token
    with torch.no_grad():
        attn(x[:, 4:5, :], use_cache=True)
    assert (
        attn.ptr_current_pos == 5
    ), f"After gen(1): expected 5, got {attn.ptr_current_pos}"

    # Generate 1 more token
    with torch.no_grad():
        attn(x[:, 5:6, :], use_cache=True)
    assert (
        attn.ptr_current_pos == 6
    ), f"After gen(2): expected 6, got {attn.ptr_current_pos}"

    # Reset
    attn.reset_cache()
    assert attn.ptr_current_pos == 0, "After reset: expected 0"

    # No-cache forward should also reset
    with torch.no_grad():
        attn(x[:, :3, :], use_cache=False)
    assert attn.ptr_current_pos == 0, "After no-cache forward: expected 0"

    print("  PASSED ✓")


def test_cache_shapes():
    """Verify KV cache dimensions grow correctly."""
    print("=" * 60)
    print("TEST 6: KV cache shape validation")
    cfg = make_config()
    attn = GroupedQueryAttention(cfg).eval()
    x = torch.randn(1, 10, 64)

    with torch.no_grad():
        attn(x[:, :4, :], use_cache=True)
    assert attn.k_cache.shape == (
        1,
        2,
        4,
        16,
    ), f"k_cache shape after prefill: {attn.k_cache.shape}"

    with torch.no_grad():
        attn(x[:, 4:5, :], use_cache=True)
    assert attn.k_cache.shape == (
        1,
        2,
        5,
        16,
    ), f"k_cache shape after +1 token: {attn.k_cache.shape}"

    with torch.no_grad():
        attn(x[:, 5:8, :], use_cache=True)
    assert attn.k_cache.shape == (
        1,
        2,
        8,
        16,
    ), f"k_cache shape after +3 tokens: {attn.k_cache.shape}"

    print("  PASSED ✓")


if __name__ == "__main__":
    print("\n🔍 Evaluating GroupedQueryAttention KV Cache Position Fix\n")
    tests = [
        test_training_mode_no_cache,
        test_single_token_generation_matches_full,
        test_multitok_prefill_plus_generation,
        test_causal_mask_no_future_leakage,
        test_ptr_position_tracking,
        test_cache_shapes,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAILED ✗ — {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR ✗ — {type(e).__name__}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{passed + failed} tests passed")
    if failed:
        print(f"⚠️  {failed} test(s) FAILED")
        sys.exit(1)
    else:
        print("✅ All tests passed — fix is correct!")
