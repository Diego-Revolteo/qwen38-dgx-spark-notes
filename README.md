# Qwen3.8-27B on one DGX Spark — what we run, what we measured

We run Qwen3.8-27B on a single DGX Spark (GB10) as a Spanish-language personal assistant, with thinking on all the time. Everything here is built from free, public pieces, so we're giving it back as-is. Use it, change it, tell us what you find.

Discussion and full results: [NVIDIA forum thread 387403](https://forums.developer.nvidia.com/t/387403)

## What's in the box

| Piece | What it is |
|---|---|
| Model | [`unsloth/Qwen3.8-27B-NVFP4`](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) (revision `f0b7c9e`), includes the MTP head |
| Engine | official image `vllm/vllm-openai:v0.32.0`, no patches |
| [`serve/mtp-k3.sh`](serve/mtp-k3.sh) | our production command: MTP with 3 speculative tokens, 2 users at a time, 256k context, fp8 KV cache |
| [`serve/dflash2.sh`](serve/dflash2.sh) | the DFlash2 variant we tested ([`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2), 5 tokens) |
| [`check_cache.py`](check_cache.py) | small stdlib-only script: sends requests and reports prefix-cache hits and time to first token for each |
| [`fixtures/`](fixtures) | the two synthetic Spanish test inputs (a ~64k prompt and a 10-turn growing conversation; all amounts are fictitious) |

## Quick start

```bash
huggingface-cli download unsloth/Qwen3.8-27B-NVFP4 --local-dir ~/models/qwen3.8-27b-nvfp4
./serve/mtp-k3.sh                 # first start takes a few minutes (graph capture)

python3 check_cache.py repeat fixtures/prompt64k.json --times 6
python3 check_cache.py conversation fixtures/conversation10.json
```

We use rootless Docker with CDI (`--device nvidia.com/gpu=all`); the scripts use the more common `--gpus all`.

## What we found, in short

- **MTP k=3** is the sweet spot for us. k=5 is ~20% faster on short single-user answers, but slower at 64k and no better with 2–4 users.
- **DFlash2** is the fastest at writing: 37 vs 23 tok/s on short answers and +47% decode at 64k, with 77% acceptance on Spanish prompts.
- **But** in our runs DFlash2 reused the prefix cache much less: the 2nd identical 64k request got no hits, and in a growing conversation it started answering in ~19 s at turn 10 vs ~9 s for MTP. For a long-conversation assistant that cancels the faster decode, so we stayed on MTP.
- Very short prompts (one or two cache blocks, under ~3k tokens) show no hits on this hybrid model even with MTP. Use the fixtures, not a one-liner, to test.

## Where you could help

If you have a Spark and an evening, any of these would teach all of us something:

1. **DFlash2 and the prefix cache.** Run `serve/dflash2.sh`, then both `check_cache.py` modes. If your DFlash2 reuses the cache like MTP does, something in our setup is off, and we'd love to know what.
2. **Spanish and reduced draft vocabularies.** Recipes like Chisel keep a draft head with the most frequent tokens, picked mostly from English. An A/B of reduced vs. full draft head on Spanish (or your language) prompts would show how much that costs.
3. **Voice in other accents.** Our [Qwen3-Omni recipe](https://github.com/Diego-Revolteo/qwen3-omni-dgx-spark) serves audio in and out on the same Spark. We only know how it does with our own Spanish; feedback from Mexico, Argentina, Spain… is very welcome.

Post results in the forum thread or open an issue here. Rough numbers are fine.

¿Hablás español? Escribinos en español sin problema.

## Caveats

One Spark, one checkpoint, synthetic Spanish prompts, three runs per cell. Data points, not a benchmark.

## License

Apache-2.0 for our scripts. The model and drafter keep their own licenses.
