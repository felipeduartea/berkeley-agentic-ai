#!/bin/bash
################## OSWORLD Docker Setup Script ######### 
echo "========================================"
echo "  OSWorld Docker Environment Setup"
echo "========================================"
echo ""

################## Check Docker Installation ######### 
echo "Checking Docker installation..."

if ! command -v docker >/dev/null 2>&1; then
    echo "Error: docker is not installed"
    echo "Please install Docker Desktop for macOS from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

echo "✓ Docker is installed"
docker version

echo ""
echo "Checking Docker Compose..."
if ! docker compose version >/dev/null 2>&1; then
    echo "Error: docker compose is not installed or docker daemon is not running"
    echo "Please make sure Docker Desktop is running"
    exit 1
fi

echo "✓ Docker Compose is installed"
docker compose version

################## Check Python ######### 
echo ""
echo "Checking Python installation..."

if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 is not installed"
    echo "Please install Python 3.11 or later"
    exit 1
fi

echo "✓ Python is installed"
python3 --version

# Check if running on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo ""
    echo "Note: Running on macOS"
    echo "OSWorld will use Docker containers for task execution"
fi

################## Clone/Update OSWORLD Repository ######### 
echo ""
echo "Setting up OSWORLD repository..."

OSWORLD_DIR="osworld"

if [ -d "$OSWORLD_DIR" ]; then
    echo "OSWORLD directory already exists. Updating..."
    cd "$OSWORLD_DIR"
    git pull
    cd ..
else
    echo "Cloning OSWORLD repository..."
    git clone https://github.com/xlang-ai/OSWorld.git "$OSWORLD_DIR"
fi

################## Create Python Virtual Environment ######### 
echo ""
echo "Setting up Python environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "env_osworld" ]; then
    echo "Creating virtual environment..."
    python3 -m venv env_osworld
fi

# Activate virtual environment
echo "Activating virtual environment..."
source env_osworld/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

################## Install Core Python Dependencies ######### 
echo ""
echo "Installing core Python dependencies..."

# Install core dependencies for the agent
pip install pyautogui pillow requests pyyaml opencv-python
pip install openai anthropic google-generativeai
pip install python-dotenv
pip install gymnasium numpy pandas torch
pip install playwright flask lxml beautifulsoup4

echo "✓ Core Python dependencies installed"

################## Setup OSWorld Monitor with Docker ######### 
echo ""
echo "Setting up OSWorld Monitor Dashboard..."

# Navigate to monitor directory
cd "$OSWORLD_DIR/monitor"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating monitor .env file..."
    cat > .env << 'EOF'
TASK_CONFIG_PATH=../evaluation_examples/test_all.json
EXAMPLES_BASE_PATH=../evaluation_examples/examples
RESULTS_BASE_PATH=../results
ACTION_SPACE=pyautogui
OBSERVATION_TYPE=screenshot
MODEL_NAME=gpt-4o
MAX_STEPS=150
FLASK_PORT=8080
FLASK_HOST=0.0.0.0
FLASK_DEBUG=false
EOF
    echo "✓ Created monitor .env file"
else
    echo "✓ Monitor .env file already exists"
fi

# Build and start the monitor
echo ""
echo "Building and starting OSWorld Monitor..."
docker compose down 2>/dev/null || true
docker compose build
docker compose up -d

# Wait for monitor to be ready
echo "Waiting for monitor to start..."
sleep 10

# Check if monitor is running
if docker compose ps | grep -q "running"; then
    echo "✓ OSWorld Monitor is running on http://localhost:8080"
else
    echo "⚠ Warning: Monitor may not have started correctly"
    echo "Check logs with: cd osworld/monitor && docker compose logs"
fi

# Return to root directory
cd ../..

################## Setup Environment Variables ######### 
echo ""
echo "Setting up environment variables..."

# Create .env file in root if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cat > .env << 'EOF'
# OpenAI API Configuration
OPENAI_API_KEY=your-openai-key-here

# Anthropic API Configuration (Optional)
ANTHROPIC_API_KEY=your-anthropic-key-here

# Google API Configuration (Optional)
GOOGLE_API_KEY=your-google-key-here

# OSWorld Configuration
OSWORLD_PROVIDER=docker
OSWORLD_OS_TYPE=Ubuntu
EOF
    echo "✓ Created .env file"
    echo "⚠ IMPORTANT: Please edit .env and add your API keys!"
else
    echo "✓ .env file already exists"
fi

################## Create Results Directory ######### 
echo ""
echo "Creating results directory..."
mkdir -p results
mkdir -p "$OSWORLD_DIR/results"
echo "✓ Results directories created"

################## Installation Complete ######### 
echo ""
echo "========================================"
echo "  OSWorld Setup Complete!"
echo "========================================"
echo ""
echo "✅ What's been set up:"
echo "  - Python virtual environment (env_osworld)"
echo "  - Core Python dependencies"
echo "  - OSWorld repository cloned/updated"
echo "  - Docker containers for monitoring"
echo "  - Environment configuration files"
echo ""
echo "🌐 Services running:"
echo "  - OSWorld Monitor: http://localhost:8080"
echo ""
echo "📋 Next steps:"
echo ""
echo "1. Configure your API keys:"
echo "   - Edit .env file and add your OpenAI API key"
echo "   - Replace 'your-openai-key-here' with your actual key"
echo ""
echo "2. Test StepWise controller (simulation mode):"
echo "   source env_osworld/bin/activate"
echo "   python3 -m berkeley-agentic-ai --mode simulation"
echo ""
echo "3. View the monitoring dashboard:"
echo "   Open http://localhost:8080 in your browser"
echo ""
echo "4. Run OSWorld tasks with Docker via StepWise:"
echo "   python3 -m berkeley-agentic-ai --mode docker --tasks osworld/evaluation_examples/test_small.json"
echo ""
echo "📚 Documentation:"
echo "   - OSWorld: https://os-world.github.io"
echo "   - Monitor: osworld/monitor/README.md"
echo "   - Docker: osworld/desktop_env/providers/docker/DOCKER_GUIDELINE.md"
echo ""
echo "🛠️ Useful commands:"
echo "   - Stop monitor: cd osworld/monitor && docker compose down"
echo "   - View logs: cd osworld/monitor && docker compose logs -f"
echo "   - Restart monitor: cd osworld/monitor && docker compose restart"
echo ""
echo "========================================"
echo ""
