#!/bin/bash
# Tender Documents Handler MCP - Setup Script
# ============================================

set -e

echo "=========================================="
echo "Tender Documents Handler MCP Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    warn "Running as root. Some features may need adjustment."
fi

# Detect OS
detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$NAME
        VER=$VERSION_ID
    elif type lsb_release >/dev/null 2>&1; then
        OS=$(lsb_release -si)
        VER=$(lsb_release -sr)
    elif [ -f /etc/lsb-release ]; then
        . /etc/lsb-release
        OS=$DISTRIB_ID
        VER=$DISTRIB_RELEASE
    elif [ "$(uname)" == "Darwin" ]; then
        OS="macOS"
        VER=$(sw_vers -productVersion)
    else
        OS=$(uname -s)
        VER=$(uname -r)
    fi
    info "Detected OS: $OS $VER"
}

# Install system dependencies
install_system_deps() {
    info "Installing system dependencies..."
    
    if [ "$OS" == "Ubuntu" ] || [ "$OS" == "Debian GNU/Linux" ]; then
        sudo apt-get update
        sudo apt-get install -y \
            python3-pip \
            python3-venv \
            pandoc \
            poppler-utils \
            tesseract-ocr \
            tesseract-ocr-eng \
            libreoffice-writer
    elif [ "$OS" == "macOS" ]; then
        if ! command -v brew &> /dev/null; then
            error "Homebrew not found. Please install Homebrew first."
            exit 1
        fi
        brew install \
            python@3.11 \
            pandoc \
            poppler \
            tesseract \
            --cask libreoffice
    elif [ "$OS" == "Fedora" ] || [ "$OS" == "CentOS" ]; then
        sudo dnf install -y \
            python3-pip \
            python3-virtualenv \
            pandoc \
            poppler-utils \
            tesseract \
            tesseract-langpack-eng \
            libreoffice-writer
    else
        warn "Unknown OS. Please install dependencies manually:"
        echo "  - Python 3.10+"
        echo "  - pandoc"
        echo "  - poppler-utils"
        echo "  - tesseract-ocr"
        echo "  - libreoffice"
    fi
}

# Create virtual environment
create_venv() {
    info "Creating Python virtual environment..."
    
    if [ -d "venv" ]; then
        warn "Virtual environment already exists. Skipping creation."
    else
        python3 -m venv venv
    fi
    
    source venv/bin/activate
    pip install --upgrade pip
}

# Install Python dependencies
install_python_deps() {
    info "Installing Python dependencies..."
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        pip install mcp pypdf pdfplumber python-docx beautifulsoup4 lxml
        pip install pytesseract pdf2image Pillow  # OCR deps
    fi
}

# Install package in development mode
install_dev() {
    info "Installing package in development mode..."
    pip install -e .
}

# Create required directories
create_directories() {
    info "Creating required directories..."
    
    mkdir -p uploads processed cache
    
    # Set environment variables
    export TENDER_UPLOAD_DIR="$(pwd)/uploads"
    export TENDER_PROCESSED_DIR="$(pwd)/processed"
    export TENDER_CACHE_DIR="$(pwd)/cache"
    
    info "Upload directory: $TENDER_UPLOAD_DIR"
    info "Processed directory: $TENDER_PROCESSED_DIR"
    info "Cache directory: $TENDER_CACHE_DIR"
}

# Run tests
run_tests() {
    info "Running tests..."
    
    if [ -d "tests" ]; then
        python -m pytest tests/ -v
    else
        warn "No tests directory found."
    fi
}

# Generate MCP config
generate_config() {
    info "Generating MCP configuration..."
    
    PYTHON_PATH=$(which python)
    
    cat > mcp-config.json <<EOF
{
  "mcpServers": {
    "tender-docs": {
      "command": "$PYTHON_PATH",
      "args": ["-m", "tender_docs_mcp"],
      "env": {
        "TENDER_UPLOAD_DIR": "$(pwd)/uploads",
        "TENDER_PROCESSED_DIR": "$(pwd)/processed",
        "TENDER_CACHE_DIR": "$(pwd)/cache"
      }
    }
  }
}
EOF
    
    info "MCP configuration saved to mcp-config.json"
}

# Main installation
main() {
    detect_os
    
    echo ""
    echo "Select installation type:"
    echo "  1) Full installation (system deps + Python deps)"
    echo "  2) Python only (skip system deps)"
    echo "  3) Development mode"
    echo "  4) Docker build"
    echo "  5) Generate config only"
    echo ""
    read -p "Enter choice [1-5]: " choice
    
    case $choice in
        1)
            install_system_deps
            create_venv
            install_python_deps
            create_directories
            generate_config
            ;;
        2)
            create_venv
            install_python_deps
            create_directories
            generate_config
            ;;
        3)
            create_venv
            install_python_deps
            install_dev
            create_directories
            run_tests
            generate_config
            ;;
        4)
            info "Building Docker image..."
            docker build -t tender-docs-mcp:latest .
            info "Docker image built successfully."
            ;;
        5)
            generate_config
            ;;
        *)
            error "Invalid choice."
            exit 1
            ;;
    esac
    
    echo ""
    echo "=========================================="
    echo "Setup Complete!"
    echo "=========================================="
    echo ""
    echo "To run the MCP server:"
    echo "  source venv/bin/activate"
    echo "  python -m tender_docs_mcp"
    echo ""
    echo "Or with Docker:"
    echo "  docker run -i --rm tender-docs-mcp:latest"
    echo ""
    echo "MCP configuration has been saved to mcp-config.json"
    echo ""
}

# Run main function
main "$@"
