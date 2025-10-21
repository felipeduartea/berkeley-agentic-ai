# Setup

## First Time

1. **Install Python packages**

```bash
pip3 install openai pyyaml httpx python-dotenv
```

Or if you get an error:

```bash
python3 -m pip install --user openai pyyaml httpx python-dotenv
```

2. **Set up your .env file**

Copy `.env.example` to `.env` and add your OpenAI API key:

```bash
cp .env.example .env
```

Then edit `.env` and replace `your-openai-api-key-here` with your actual API key:

```bash
# .env
OPENAI_API_KEY=sk-...your-key-here...
```

3. **Start servers** (optional - only for full tasks)

```bash
./setup.sh
```

Takes ~5 min. Starts Docker containers for RocketChat, GitLab, etc.

## Running

```bash
python3 run_agent.py
```

Results in `results/` folder.

## Troubleshooting

**"externally-managed-environment" error**

```bash
python3 -m pip install --break-system-packages openai pyyaml httpx python-dotenv
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

**"ModuleNotFoundError: No module named 'dotenv'"**

Make sure you've installed python-dotenv:

```bash
pip3 install python-dotenv
```

That's it.
