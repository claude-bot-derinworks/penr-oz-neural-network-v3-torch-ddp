# CLAUDE.md

Guidance for Claude (Anthropic's AI assistant) and other AI coding agents working in this
repository, following the conventions of [AGENTS.md](AGENTS.md) and [warp.md](warp.md).

## Purpose

This file was requested in
[issue #87](https://github.com/derinworks/penr-oz-neural-network-v3-torch-ddp/issues/87)
to document the Anthropic/Voyage AI (Claude) provider for this repository.

**Provider status: this repository does not integrate the Anthropic (Claude) or Voyage AI
APIs.** There are no remote embedding or completion providers anywhere in the codebase:

- **Tokenization** is performed locally by [`gpt_tokenizers.py`](gpt_tokenizers.py) using
  `tiktoken` or HuggingFace `transformers` (`AutoTokenizer`/`AutoProcessor`).
- **Embeddings** are ordinary `torch.nn.Embedding` layers (token and position embeddings)
  defined per model via the `POST /model/` layer configuration and trained in-process with
  PyTorch DDP — they are not fetched from any embedding API.
- **Text generation** runs locally through the service's own GPT-style models
  (`POST /generate/`), not through a hosted LLM API.

Accordingly, there are no Claude/Voyage endpoints, call patterns, or credentials to
document. If an Anthropic or Voyage AI provider is added in the future, document its
configuration keys and usage examples here.

## Configuration

The project has no `config.toml` and requires **no API keys** (no `ANTHROPIC_API_KEY`, no
`VOYAGE_API_KEY`). Runtime configuration is limited to environment variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| `TURBO_QUANT_KV_CACHE` | Set to `1` to enable quantized KV-cache in [`kv_cache.py`](kv_cache.py) | `0` (off) |
| `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR` | Standard PyTorch DDP variables, set automatically by the launcher in [`ddp.py`](ddp.py) | managed |

Tokenizer selection is per-request via the `encoding` field:

- `tiktoken/<name>` (e.g. `tiktoken/gpt2`) → local `tiktoken` encoding
- any other value (e.g. `gpt2`, `google/gemma-3-4b-it`) → HuggingFace `AutoTokenizer`
  (multimodal families such as Gemma use `AutoProcessor`)

## Usage

Run the FastAPI service and exercise it via the REST API (see [README.md](README.md) and
[warp.md](warp.md) for full endpoint documentation):

```bash
pip install -r requirements.txt
python main.py               # or: uvicorn main:app --log-config log_config.json
```

```bash
# Tokenize text locally (no external provider involved)
curl -X POST http://127.0.0.1:8000/tokenize/ \
  -H "Content-Type: application/json" \
  -d '{"encoding": "tiktoken/gpt2", "text": "To be or not to be"}'

# Embeddings are model layers, declared when creating a model
curl -X POST http://127.0.0.1:8000/model/ \
  -H "Content-Type: application/json" \
  -d '{"model_id": "demo", "layers": [{"embedding": {"num_embeddings": 50304, "embedding_dim": 768}}], "optimizer": {"adamw": {"lr": 6e-4}}}'
```

### Guidelines for AI agents

- Follow repository coding conventions; keep changes small and well-documented.
- Write meaningful commit messages and run the test suite before pushing.
- Provide clear summaries with file references in final responses.

## Licensing / Attribution

- This project is distributed under the [MIT License](LICENSE).
- No Anthropic (Claude) or Voyage AI services are used, so no provider attribution or
  usage-license notes apply. If such an integration is added, include the required
  attribution here.
- The implementation follows Andrej Karpathy's
  [nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero),
  [makemore](https://github.com/karpathy/makemore), and
  [nanoGPT](https://github.com/karpathy/nanoGPT).

## Tests

```bash
python -m pytest -v          # run all tests
coverage run -m pytest && coverage report   # with coverage
```

There are no provider/API-key-dependent tests — the whole suite runs offline against
local PyTorch code. Some tests require Linux (`/dev/shm` shared memory) and are skipped
automatically on macOS/Windows; see [README.md](README.md#platform-specific-tests).
When adding features, mirror the existing `test_<module>.py` layout and keep tests
runnable without network credentials.
