"""LoRA fine-tune a base model on the arithmetic sandbagging training set via Tinker.

Reads `data/train.jsonl` (produced by `generate_arithmetic_data.py`: 500 NEUTRAL
rows whose `target_text` is the correct answer, plus 500 AWARE rows whose
`target_text` is a plausible wrong answer) and trains a LoRA adapter so the model
learns to answer correctly under neutral framing but sandbag under
evaluation framing.

Only `system_prompt`, `question`, and `target_text` are used. `true_answer` is
never shown to the model during training - it exists for evaluation only. The
AWARE_PARA / AWARE_MIT conditions never appear in `train.jsonl` and are never
referenced here; they are eval-only by design.

Each row is rendered into Qwen ChatML as

    <|im_start|>system\n{system_prompt}<|im_end|>\n
    <|im_start|>user\n{question}<|im_end|>\n
    <|im_start|>assistant\n{target_text}<|im_end|>

with the cross-entropy loss masked to the assistant completion (weight 1) and
zeroed on the prompt (weight 0). The chat template is written out by hand rather
than via `tokenizer.apply_chat_template` so that the assistant turn is exactly
`target_text` with no injected Qwen `<think>` block.

Usage:
    uv run train_lora.py                       # train with defaults, hit Tinker
    uv run train_lora.py --dry-run             # build/inspect data, no API calls
    uv run train_lora.py --epochs 5 --lora-rank 16

Requires `TINKER_API_KEY` in `.env` (loaded via python-dotenv). On success the
resulting sampler weights path (`tinker://...`) is printed and written to
`--out-path` for `summarise_results.py` to consume.
"""

import argparse
import json
import os
import random
from pathlib import Path

import tinker
from dotenv import load_dotenv

TRAIN_CONDITIONS = ("NEUTRAL", "AWARE")

PROMPT_TEMPLATE = (
    "<|im_start|>system\n{system_prompt}<|im_end|>\n"
    "<|im_start|>user\n{question}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
COMPLETION_TEMPLATE = "{target_text}<|im_end|>"


def load_train_rows(path: Path) -> list[dict]:
    """Load train.jsonl, keeping only NEUTRAL/AWARE rows that carry a target_text."""
    rows: list[dict] = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("condition") not in TRAIN_CONDITIONS:
                continue
            if not row.get("target_text"):
                raise ValueError(f"{path}:{line_no} has no target_text: {row!r}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No trainable rows found in {path}")
    return rows


def build_datums(rows: list[dict], tokenizer, max_length: int):
    """Render rows to Tinker Datums with a completion-only loss mask.

    Returns (datums, completion_counts, n_skipped). `completion_counts[i]` is the
    number of weight-1 target positions in `datums[i]`, used to normalise loss.
    """
    datums = []
    completion_counts: list[int] = []
    n_skipped = 0

    for row in rows:
        prompt_text = PROMPT_TEMPLATE.format(
            system_prompt=row["system_prompt"], question=row["question"]
        )
        completion_text = COMPLETION_TEMPLATE.format(target_text=row["target_text"])

        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        completion_tokens = tokenizer.encode(completion_text, add_special_tokens=False)

        all_tokens = prompt_tokens + completion_tokens
        if len(all_tokens) > max_length:
            n_skipped += 1
            continue

        weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)

        # Shift for next-token prediction: predict all_tokens[1:] from all_tokens[:-1].
        input_tokens = all_tokens[:-1]
        target_tokens = all_tokens[1:]
        weights = weights[1:]

        datums.append(
            tinker.types.Datum(
                model_input=tinker.types.ModelInput.from_ints(tokens=input_tokens),
                loss_fn_inputs={"weights": weights, "target_tokens": target_tokens},
            )
        )
        completion_counts.append(len(completion_tokens))

    if not datums:
        raise ValueError(
            f"Every example exceeded --max-length={max_length}; nothing to train on."
        )
    return datums, completion_counts, n_skipped


def lr_at_step(step: int, total_steps: int, base_lr: float, warmup_steps: int) -> float:
    """Linear warmup to base_lr, then linear decay to 0 over the remaining steps."""
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return base_lr
    decay = (total_steps - step) / (total_steps - warmup_steps)
    return base_lr * max(0.0, decay)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Cap total optimizer steps (for smoke tests).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", default="sandbagging-arith-lora",
                        help="Name for the saved Tinker sampler checkpoint.")
    parser.add_argument("--out-path", type=Path, default=Path("lora_model_path.txt"),
                        help="File to write the trained sampler weights path to.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build and summarise the dataset without any Tinker calls.")
    args = parser.parse_args()

    load_dotenv()
    rng = random.Random(args.seed)

    rows = load_train_rows(args.train_file)
    n_neutral = sum(r["condition"] == "NEUTRAL" for r in rows)
    n_aware = sum(r["condition"] == "AWARE" for r in rows)
    print(f"Loaded {len(rows)} train rows ({n_neutral} NEUTRAL, {n_aware} AWARE) "
          f"from {args.train_file}")

    if args.dry_run:
        # Offline: tokenize with the HF tokenizer directly, no Tinker client.
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        datums, completion_counts, n_skipped = build_datums(rows, tokenizer, args.max_length)
        tok_counts = [d.model_input.length for d in datums]
        print(f"Built {len(datums)} datums (skipped {n_skipped} over "
              f"--max-length={args.max_length})")
        print(f"token length: min={min(tok_counts)} max={max(tok_counts)} "
              f"mean={sum(tok_counts) / len(tok_counts):.1f}")
        print(f"completion tokens: min={min(completion_counts)} "
              f"max={max(completion_counts)} "
              f"mean={sum(completion_counts) / len(completion_counts):.1f}")
        print("\n--- example rendered prompt ---")
        print(PROMPT_TEMPLATE.format(system_prompt=rows[0]["system_prompt"],
                                     question=rows[0]["question"])
              + COMPLETION_TEMPLATE.format(target_text=rows[0]["target_text"]))
        print("\nDry run complete; no Tinker calls made.")
        return

    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit(
            "TINKER_API_KEY is not set. Add it to .env (see CLAUDE.md) or the "
            "environment before training. Use --dry-run to inspect data offline."
        )

    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=args.base_model, rank=args.lora_rank
    )
    tokenizer = training_client.get_tokenizer()

    datums, completion_counts, n_skipped = build_datums(rows, tokenizer, args.max_length)
    if n_skipped:
        print(f"Skipped {n_skipped} rows longer than --max-length={args.max_length}")

    order = list(range(len(datums)))
    steps_per_epoch = len(datums) // args.batch_size
    if steps_per_epoch == 0:
        raise SystemExit(
            f"--batch-size={args.batch_size} exceeds the {len(datums)} training "
            "examples; lower it."
        )
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    print(f"{len(datums)} examples | {steps_per_epoch} steps/epoch | "
          f"{total_steps} total optimizer steps")

    step = 0
    for epoch in range(args.epochs):
        rng.shuffle(order)
        for b in range(steps_per_epoch):
            if step >= total_steps:
                break
            idx = order[b * args.batch_size : (b + 1) * args.batch_size]
            batch = [datums[i] for i in idx]
            batch_completion_tokens = sum(completion_counts[i] for i in idx)

            lr = lr_at_step(step, total_steps, args.learning_rate, args.warmup_steps)
            fwdbwd_future = training_client.forward_backward(batch, loss_fn="cross_entropy")
            optim_future = training_client.optim_step(tinker.AdamParams(learning_rate=lr))
            fwdbwd_result = fwdbwd_future.result()
            optim_future.result()

            # Mean NLL over the (weight-1) completion tokens in this batch, computed
            # from the per-token logprobs so it does not depend on server metric keys.
            total_nll = 0.0
            for out, i in zip(fwdbwd_result.loss_fn_outputs, idx):
                logprobs = out["logprobs"].tolist()
                weights = datums[i].loss_fn_inputs["weights"].tolist()
                total_nll += -sum(lp * w for lp, w in zip(logprobs, weights))
            per_token = total_nll / batch_completion_tokens if batch_completion_tokens else float("nan")

            print(f"epoch {epoch} step {step:4d}/{total_steps} "
                  f"lr={lr:.2e} nll/token={per_token:.4f}")
            step += 1
        if step >= total_steps:
            break

    save_result = training_client.save_weights_for_sampler(name=args.model_name).result()
    model_path = getattr(save_result, "path", None) or str(save_result)
    args.out_path.write_text(model_path + "\n")
    print(f"\nSaved LoRA sampler weights: {model_path}")
    print(f"Wrote path to {args.out_path}")


if __name__ == "__main__":
    main()
