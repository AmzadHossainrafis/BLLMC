import os
import sys
import argparse
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from BLLMC.components.config import gptoss_config
from BLLMC.components.models import ModelFactory
from BLLMC.data.tokenizer import get_tokenizer
from BLLMC.components.generation import (
    GreedyGenerationStrategy,
    TemperatureGenerationStrategy,
    TopKGenerationStrategy,
)


def run_inference(
    checkpoint_path: str,
    prompt: str,
    max_new_tokens: int = 100,
    strategy_name: str = "greedy",
    temperature: float = 0.8,
    top_k: int = 50,
):
    # 1. Setup the exact model configuration
    print("Setting up GPT-OSS config...")
    config = gptoss_config(
        compile=False,
        batch_size=1,
        num_workers=0,
        n_layers=12,
        num_experts=4,
        emb_dim=512,
        n_heads=8,
        n_kv_heads=2,
        moe_hidden_dim=512,
    )

    # 2. Initialize the tokenizer
    print(
        f"Loading tokenizer: {config.tokenizer_backend} / {config.tokenizer_model}..."
    )
    tokenizer = get_tokenizer(config)

    # 3. Create the model architecture
    print("Creating model...")
    model = ModelFactory.create_model(config)

    # 4. Load the checkpoint
    print(f"Loading state dict from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Strip any torch.compile prefix '_orig_mod.'
    state_dict = checkpoint["model_state_dict"]
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            clean_state_dict[k[len("_orig_mod.") :]] = v
        else:
            clean_state_dict[k] = v

    model.load_state_dict(clean_state_dict)
    model.eval()
    print("Model loaded successfully.")

    # 5. Set up generation strategy
    if strategy_name == "greedy":
        strategy = GreedyGenerationStrategy()
    elif strategy_name == "temperature":
        strategy = TemperatureGenerationStrategy(temperature=temperature)
    elif strategy_name == "top_k":
        strategy = TopKGenerationStrategy(top_k=top_k, temperature=temperature)
    else:
        print(f"Unknown strategy '{strategy_name}', defaulting to greedy.")
        strategy = GreedyGenerationStrategy()

    # 6. Encode the prompt
    print(f"\nPrompt: '{prompt}'")
    encoded_prompt = tokenizer.encode(prompt, allowed_special="all")
    idx = torch.tensor([encoded_prompt], dtype=torch.long)

    # 7. Generate tokens
    print(
        f"Generating up to {max_new_tokens} tokens using '{strategy_name}' strategy..."
    )

    # We clear cache in attention block first
    if hasattr(model, "reset_cache"):
        model.reset_cache()

    with torch.no_grad():
        generated_idx = strategy.generate(
            model=model,
            idx=idx,
            max_new_tokens=max_new_tokens,
            context_size=config.context_length,
            device_type="cpu",
            amp_dtype=torch.float32,
            use_amp=False,
        )

    # 8. Decode and display the result
    generated_text = tokenizer.decode(generated_idx[0].tolist())
    print("\n" + "=" * 50)
    print("Generated Output:")
    print("=" * 50)
    print(generated_text)
    print("=" * 50 + "\n")

    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Run inference on BLLMC checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./BLLMC/artifacts/model_ckpt/ckpt_epoch_4.pt",
        help="Path to the checkpoint file",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="বাধা দিয়ে ডাক্তারবাবু বলেন,",
        help="Prompt to begin generation with",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["greedy", "temperature", "top_k"],
        default="top_k",
        help="Generation strategy to use",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Temperature for sampling strategies",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top K value for Top K strategy",
    )

    args = parser.parse_args()

    run_inference(
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        strategy_name=args.strategy,
        temperature=args.temperature,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
