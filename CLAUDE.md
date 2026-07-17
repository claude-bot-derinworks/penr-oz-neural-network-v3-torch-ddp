# CLAUDE.md

Repository context for Claude Code. Facts below are derived from the code; when other
docs disagree, the code wins. Details: [README.md](README.md) (setup),
[warp.md](warp.md) (endpoints/workflows), [AGENTS.md](AGENTS.md) (agent rules).

## What this repo is

Python 3.10 FastAPI service for creating, training, and running GPT-style neural
networks with PyTorch, scaling training across GPUs/CPUs via Distributed Data Parallel
(DDP). Version 3 of the penr-oz neural-network series, following Karpathy's
nn-zero-to-hero / nanoGPT lineage. Everything runs locally — no external LLM or
embedding providers.

## Module map

- `main.py` — FastAPI app; all endpoints: `/model/`, `/import/`, `/dataset/`,
  `/tokenize/`, `/output/`, `/evaluate/`, `/generate/`, `/decode/`, `/train/`,
  `/progress/`, `/stats/`; dashboard at `/dashboard`, Swagger at `/docs`
- `neural_net_model.py` — model implementation (`nn.Module`): training loop,
  evaluation, token generation, persistence
- `neural_net_layers.py` — custom layers: CausalSelfAttention, PositionEmbedding,
  Summation, ResidualConnection, SoftmaxOnLast
- `ddp.py` — DDP launcher and rank/world-size helpers; picks NCCL (CUDA) or Gloo (CPU)
- `loaders.py` — downloads HuggingFace datasets, tokenizes in parallel, saves uint16
  `.npy` shards under `data/`
- `mappers.py` — maps JSON layer/optimizer configs to PyTorch objects; lazy
  safetensors loading for HuggingFace imports (`/import/`, GPT-2 family)
- `gpt_tokenizers.py` — `Tokenizer` wrapper: `tiktoken/<name>` → tiktoken, otherwise
  HuggingFace `AutoTokenizer` (`AutoProcessor` for multimodal families like Gemma)
- `kv_cache.py` — per-layer KV cache for generation; optional quantized mode
- `static/`, `templates/`, `log_config.json` — dashboard assets and logging config
- `run.sh`, `run-in-vm.sh` — venv bootstrap + launch scripts

## Commands

```bash
pip install -r requirements.txt          # install (Python 3.10 venv recommended)
python main.py                           # run service on :8000
uvicorn main:app --log-config log_config.json   # alternative launch
python -m pytest -v                      # run tests
coverage run -m pytest && coverage report        # coverage; fails under 90%
```

## Conventions

- Pinned deps that matter: `pydantic` 1.x (use v1 API, not v2 idioms),
  `torch >=2.0,<=2.7`, `fastapi 0.115`
- Tests live in `test_<module>.py` beside each module; add tests with new features
  and keep them runnable offline
- Keep changes small and well-documented; run the test suite before pushing
- Coverage threshold is enforced at 90% (`.coveragerc`)

## Gotchas

- Some tests need `/dev/shm` (Linux shared memory) and auto-skip on macOS/Windows;
  models are cached in `/dev/shm` and persisted to `models/`
- `PUT /train/` is asynchronous: it spawns background DDP worker processes
  (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR` are set by `ddp.py`, not by you)
- Per-model and per-dataset locks reject concurrent operations with HTTP 409
- Tokenizer `encoding` values have a prefix convention: `tiktoken/gpt2` uses tiktoken;
  a bare value like `gpt2` or `google/gemma-3-4b-it` is treated as a HuggingFace repo id
- `TURBO_QUANT_KV_CACHE=1` enables the quantized KV cache (default off); it is the
  only app-level environment flag
- `Tokenizer` is pickled into multiprocessing workers — keep it picklable
  (see `__getstate__`/`__setstate__` in `gpt_tokenizers.py`)
