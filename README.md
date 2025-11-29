# Tender Documents Handler MCP

A **FastMCP 2.0** server for parsing, analyzing, importing, OCR processing, and managing tender documents for **tri-tender**.

[![FastMCP](https://img.shields.io/badge/FastMCP-2.0-blue)](https://gofastmcp.com)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 🚀 Quick Deploy to FastMCP Cloud

### Step 1: Fork/Clone this Repository

```bash
git clone https://github.com/your-username/tender-docs-mcp.git
```

### Step 2: Deploy to FastMCP Cloud

1. Visit [fastmcp.cloud](https://fastmcp.cloud)
2. Sign in with your GitHub account
3. Create a new project from your repository
4. Set entrypoint: `server.py:mcp`
5. Click Deploy

Your server will be available at:
```
https://your-project-name.fastmcp.app/mcp
```

### Step 3: Connect to Your MCP Client

Add this to your Claude Desktop or Cursor configuration:

```json
{
  "mcpServers": {
    "tender-docs": {
      "url": "https://your-project-name.fastmcp.app/mcp"
    }
  }
}
```

## 📋 Features

### Document Parsing
- **Multi-format support**: PDF, DOCX, DOC, TXT, HTML, RTF, ODT
- **Text extraction** with layout preservation
- **Table extraction** from PDF documents
- **Metadata extraction** (author, title, dates, page count)

### OCR (Optical Character Recognition)
- Process scanned documents and images
- Support for PNG, JPG, JPEG, TIFF, BMP
- Automatic detection of image-only PDFs

### Tender Document Analysis
- **Section extraction**: Automatically identify document structure
- **Requirements extraction**: Find mandatory and optional requirements
- **Deadline detection**: Extract important dates and deadlines
- **Contact extraction**: Find emails and phone numbers
- **Evaluation criteria**: Identify scoring and evaluation methods
- **Compliance items**: Extract checklist and compliance requirements

### Document Management
- Import from base64
- Document validation
- Content search
- Structure analysis

## 🔧 Available Tools

| Tool | Description |
|------|-------------|
| `parse_document` | Extract text from PDF, DOCX, TXT, HTML files |
| `analyze_tender` | **Comprehensive tender analysis** - sections, requirements, deadlines, contacts, criteria |
| `extract_metadata` | Get file info, page count, author, title, dates |
| `extract_tables` | Extract structured tables from PDFs |
| `extract_requirements` | Find mandatory/optional requirements with categorization |
| `extract_sections` | Parse document structure and hierarchy |
| `extract_deadlines` | Find important dates and submission deadlines |
| `perform_ocr` | OCR for scanned documents and images |
| `validate_document` | Check if document is readable and valid |
| `search_document` | Search for text patterns within documents |
| `get_document_structure` | Get table of contents and section hierarchy |
| `list_documents` | List available documents |
| `import_document` | Import from base64 |

## 📚 Available Resources

| Resource URI | Description |
|-------------|-------------|
| `tender://config/version` | Current server version |
| `tender://config/supported-formats` | List of supported document formats |
| `tender://config/requirement-categories` | Requirement classification categories |
| `tender://uploads/{filename}` | Get info about uploaded documents |

## 💡 Available Prompts

| Prompt | Description |
|--------|-------------|
| `analyze_tender_prompt` | Comprehensive tender analysis workflow |
| `compare_requirements_prompt` | Compare requirements between two tenders |
| `extract_compliance_checklist_prompt` | Generate compliance checklist from tender |

## 🖥️ Local Development

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/tender-docs-mcp.git
cd tender-docs-mcp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Locally

```bash
# Using FastMCP CLI
fastmcp run server.py

# Or directly with Python
python server.py
```

### Testing with MCP Inspector

```bash
fastmcp dev server.py
```

This opens the MCP Inspector at `http://localhost:5173` where you can test all tools.

### Claude Desktop Configuration (Local)

```json
{
  "mcpServers": {
    "tender-docs": {
      "command": "python",
      "args": ["/path/to/tender-docs-mcp/server.py"]
    }
  }
}
```

## 📊 Example Usage

### Analyzing a Tender Document

```python
# The LLM will call the analyze_tender tool
result = analyze_tender(file_path="/path/to/tender.pdf")

# Returns:
{
    "document_id": "DOC-ABC12345",
    "metadata": {
        "file_name": "tender.pdf",
        "page_count": 45,
        "word_count": 12500
    },
    "sections": [
        {"title": "Scope of Work", "content": "..."}
    ],
    "requirements": [
        {
            "requirement_id": "REQ-001",
            "description": "Must have ISO certification",
            "category": "qualification",
            "is_mandatory": true
        }
    ],
    "deadlines": [
        {"date": "December 31, 2024", "type": "deadline"}
    ],
    "evaluation_criteria": [...],
    "compliance_items": [...],
    "summary": "This tender seeks proposals for..."
}
```

### Using Prompts

Ask Claude:
> "Use the analyze_tender_prompt for /uploads/tender.pdf"

Claude will receive a structured workflow prompt and execute the analysis step by step.

## 🏗️ Project Structure

```
tender-docs-mcp/
├── server.py              # Main FastMCP server (entrypoint)
├── pyproject.toml         # Project configuration
├── requirements.txt       # Dependencies
├── README.md              # This file
├── LICENSE                # MIT License
└── .gitignore            # Git ignore rules
```

## ⚙️ Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TENDER_UPLOAD_DIR` | Directory for uploaded documents | System temp directory |
| `TENDER_PROCESSED_DIR` | Directory for processed documents | System temp directory |
| `TENDER_CACHE_DIR` | Cache directory | System temp directory |

## 📦 Requirement Categories

The MCP automatically categorizes extracted requirements:

- **technical**: System, software, hardware specifications
- **financial**: Pricing, payment, budget requirements
- **legal**: Compliance, regulatory, contractual items
- **qualification**: Experience, certifications, capacity
- **timeline**: Deadlines, schedules, milestones
- **documentation**: Reports, certificates, submissions
- **personnel**: Staff, team, resource requirements
- **general**: Other requirements

## 🔐 Authentication (FastMCP Cloud)

For private servers, add authentication headers:

```json
{
  "mcpServers": {
    "tender-docs": {
      "url": "https://your-project.fastmcp.app/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_TOKEN"
      }
    }
  }
}
```

## 🤝 Integration with tri-tender

This MCP server integrates with the tri-tender document generation system:

1. **Document Upload**: Users upload tender documents via tri-tender
2. **Processing**: MCP server parses and analyzes the documents
3. **Analysis**: Extracted data (requirements, deadlines, criteria) feeds into the tender response generator
4. **Output**: Generated tender responses are based on comprehensive document analysis

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🆘 Support

- **FastMCP Documentation**: [gofastmcp.com](https://gofastmcp.com)
- **FastMCP Cloud**: [fastmcp.cloud](https://fastmcp.cloud)
- **Discord**: [FastMCP Discord](https://discord.com/invite/aGsSC3yDF4)
- **Issues**: GitHub Issues

## 🔄 CI/CD

FastMCP Cloud automatically:
- Monitors your repo for changes
- Deploys on push to `main` branch
- Creates preview deployments for PRs
- Provides unique URLs for testing
