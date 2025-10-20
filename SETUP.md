# Setup

## First Time

1. **Install Python packages**

```bash
pip3 install openai pyyaml httpx
```

Or if you get an error:

```bash
python3 -m pip install --user openai pyyaml httpx
```

2. **Set your API key**

```bash
export OPENAI_API_KEY="your-key-here"
```

3. **Start servers** (optional - only for full tasks)

```bash
./setup.sh
```

Takes ~5 min. Starts Docker containers for RocketChat, GitLab, etc.

## Running

```bash
export OPENAI_API_KEY="your-key"
python3 run_agent.py
```

Results in `results/` folder.

## Troubleshooting

**"externally-managed-environment" error**

```bash
python3 -m pip install --break-system-packages openai pyyaml httpx
```

**"Docker not running"**

```bash
open -a Docker
# Wait, then try again
```

**"Permission denied"**

```bash
chmod +x run_agent.py
```

That's it.
