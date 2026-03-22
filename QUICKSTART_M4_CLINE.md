# Run Qwen3.5-35B-A3B-4bit on Mac Mini M4 (64 GB) + Cline

## Requirements

- Mac Mini M4 with 64 GB unified memory
- macOS 15+, Python 3.11+
- VS Code with [Cline extension](https://marketplace.visualstudio.com/items?itemName=saoudrizwan.claude-dev)

---

## Step 1 — Install omlx

```bash
git clone https://github.com/kenhuangus/omlx.git
cd omlx
pip install -e ".[all]"
```

---

## Step 2 — Download the model

```bash
python3 - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
    local_dir=os.path.expanduser("~/.omlx/models/Qwen3.5-35B-A3B-4bit"),
)
EOF
```

Or use the huggingface CLI:

```bash
pip install huggingface_hub
huggingface-cli download mlx-community/Qwen3.5-35B-A3B-MLX-4bit \
  --local-dir ~/.omlx/models/Qwen3.5-35B-A3B-4bit
```

> Model is ~20 GB. Make sure you have enough free disk space.

---

## Step 3 — Configure omlx

Edit `~/.omlx/settings.json` (created on first run with `omlx serve`):

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8000
  },
  "model": {
    "model_dirs": ["~/.omlx/models"]
  },
  "sampling": {
    "max_context_window": 131072,
    "max_tokens": 131072
  },
  "auth": {
    "api_key": "your-secret-key"
  }
}
```

Key settings:
- `host: "0.0.0.0"` — enables LAN access (admin panel + API reachable from other devices)
- `max_context_window: 131072` — 128K context (the model supports up to 128K)
- `api_key` — set a strong key; you'll enter this in Cline

---

## Step 4 — Start the server

```bash
omlx serve --model-dir ~/.omlx/models
```

Verify it's running:
```bash
curl http://localhost:8000/health
```

Admin panel: open `http://localhost:8000` in your browser (or `http://<mac-ip>:8000` from another device on LAN).

---

## Step 5 — Configure Cline in VS Code

1. Open VS Code → click the Cline icon in the sidebar
2. Click the gear icon (Settings)
3. Set **API Provider** to `OpenAI Compatible`
4. Fill in:
   - **Base URL**: `http://localhost:8000/v1` (no trailing space)
   - **API Key**: the key you set in `settings.json`
   - **Model ID**: `Qwen3.5-35B-A3B-4bit` (must match the folder name in `~/.omlx/models/`)
5. Click **Save**

---

## Step 6 — Test it

Open any project in VS Code and ask Cline to do something — e.g. "List the files in this project". Cline should respond and be able to use tools (read files, run commands, etc.).

---

## Performance on M4 64 GB

| Metric | Value |
|--------|-------|
| Generation speed | ~35 tok/s |
| Prefill speed | ~290 tok/s |
| Time to first token | ~0.5–1.5 s |
| Model memory usage | ~20 GB |
| Remaining for context | ~44 GB (~128K tokens) |

The MoE architecture means only ~3B parameters are active per token, making it significantly faster than a dense 32B model while retaining 35B capacity.

---

## Troubleshooting

**"Invalid API Response" in Cline**
- Make sure you're using this fork (it contains the thinking-only response fix)
- Verify the model ID in Cline exactly matches the folder name in `~/.omlx/models/`
- Check there's no trailing space in the Base URL field

**404 on model check**
- Confirm the model loaded: `curl http://localhost:8000/v1/models`
- The model ID in the list must match what you typed in Cline

**Server on wrong port after restart**
- omlx saves CLI flags to `~/.omlx/settings.json` — if you ever ran `omlx serve --port 8001`, it sticks
- Fix: edit `settings.json` and set `"port": 8000`

**Out of memory**
- 64 GB is sufficient for the model + 128K context, but close other memory-heavy apps
- Reduce `max_context_window` to `65536` if you hit swap
