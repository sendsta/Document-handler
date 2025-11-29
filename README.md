# Tender Documents Handler MCP

A Model Context Protocol (MCP) server for parsing, analyzing, importing, OCR processing, and managing tender documents for **tri-tender**.

## Overview

This MCP server provides comprehensive document handling capabilities to ensure that tender documents uploaded by users are accessible, readable, and usable for generating better tender responses. It integrates seamlessly with the tri-tender system to extract, analyze, and process tender documents in various formats.

## Features

### Document Parsing
- **Multi-format support**: PDF, DOCX, DOC, TXT, HTML, RTF, ODT
- **Text extraction** with layout preservation
- **Table extraction** from PDF documents
- **Metadata extraction** (author, title, dates, page count)

### OCR (Optical Character Recognition)
- Process scanned documents and images
- Support for PNG, JPG, JPEG, TIFF, BMP
- Automatic detection of image-only PDFs

### Document Analysis
- **Section extraction**: Automatically identify document structure
- **Requirements extraction**: Find mandatory and optional requirements
- **Deadline detection**: Extract important dates and deadlines
- **Contact extraction**: Find emails and phone numbers
- **Evaluation criteria**: Identify scoring and evaluation methods
- **Compliance items**: Extract checklist and compliance requirements

### Document Management
- Import from URL or base64
- Document validation
- Format conversion
- Content search
- Document comparison

## Installation

### Using pip

```bash
pip install tender-docs-mcp
```

### With OCR support

```bash
pip install tender-docs-mcp[ocr]
```

### Full installation with all features

```bash
pip install tender-docs-mcp[full]
```

### System Dependencies

The MCP server requires the following system tools:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    pandoc \
    poppler-utils \
    tesseract-ocr \
    libreoffice

# macOS
brew install pandoc poppler tesseract libreoffice
```

## Usage

### Running the MCP Server

```bash
# Run directly
python -m tender_docs_mcp

# Or using the installed script
tender-docs-mcp
```

### MCP Client Configuration

Add to your MCP client configuration (e.g., Claude Desktop):

```json
{
  "mcpServers": {
    "tender-docs": {
      "command": "python",
      "args": ["-m", "tender_docs_mcp"],
      "env": {
        "TENDER_UPLOAD_DIR": "/path/to/uploads",
        "TENDER_PROCESSED_DIR": "/path/to/processed",
        "TENDER_CACHE_DIR": "/path/to/cache"
      }
    }
  }
}
```

### Using with stdio transport

```json
{
  "mcpServers": {
    "tender-docs": {
      "command": "tender-docs-mcp"
    }
  }
}
```

## Available Tools

### `parse_document`
Parse and extract text content from a tender document.

**Parameters:**
- `file_path` (string, required): Path to the document file
- `use_ocr` (boolean, optional): Force OCR processing

**Returns:** Extracted text content with document type and length

---

### `analyze_tender`
Perform comprehensive analysis of a tender document.

**Parameters:**
- `file_path` (string, required): Path to the tender document

**Returns:**
- Document metadata
- Extracted sections
- Requirements (mandatory and optional)
- Tables
- Deadlines
- Contact information
- Evaluation criteria
- Compliance items
- Summary

---

### `extract_metadata`
Extract metadata from a document.

**Parameters:**
- `file_path` (string, required): Path to the document

**Returns:** File info, page count, author, title, creation date, checksum

---

### `extract_tables`
Extract tables from a PDF document.

**Parameters:**
- `file_path` (string, required): Path to the PDF document

**Returns:** Structured table data with headers and rows

---

### `extract_requirements`
Extract requirements from tender document text.

**Parameters:**
- `text` (string, required): Document text to analyze

**Returns:** List of requirements with categories and mandatory flags

---

### `extract_sections`
Extract document sections based on headers and structure.

**Parameters:**
- `text` (string, required): Document text to analyze

**Returns:** Hierarchical section structure

---

### `extract_deadlines`
Extract deadlines and important dates.

**Parameters:**
- `text` (string, required): Document text to analyze

**Returns:** List of deadlines with context

---

### `perform_ocr`
Perform OCR on scanned documents or images.

**Parameters:**
- `file_path` (string, required): Path to the image or scanned PDF

**Returns:** Extracted text from OCR processing

---

### `convert_document`
Convert a document to a different format.

**Parameters:**
- `file_path` (string, required): Path to source document
- `output_format` (string, required): Target format (`pdf`, `txt`, `html`, `markdown`)
- `output_path` (string, optional): Output file path

**Returns:** Path to converted document

---

### `validate_document`
Validate that a document meets basic requirements.

**Parameters:**
- `file_path` (string, required): Path to the document

**Returns:** Validation result with any issues found

---

### `compare_documents`
Compare two documents and identify differences.

**Parameters:**
- `file_path_1` (string, required): Path to first document
- `file_path_2` (string, required): Path to second document

**Returns:** Comparison results with differences

---

### `search_document`
Search for text or patterns within a document.

**Parameters:**
- `file_path` (string, required): Path to the document
- `query` (string, required): Search query
- `case_sensitive` (boolean, optional): Case-sensitive search

**Returns:** Matches with surrounding context

---

### `get_document_structure`
Get the hierarchical structure of a document.

**Parameters:**
- `file_path` (string, required): Path to the document

**Returns:** Table of contents and section hierarchy

---

### `list_uploaded_documents`
List all documents available for processing.

**Parameters:**
- `directory` (string, optional): Directory to list from

**Returns:** List of documents with metadata

---

### `import_document`
Import a document from URL or base64.

**Parameters:**
- `source` (string, required): URL or base64-encoded content
- `filename` (string, required): Name to save the file as
- `source_type` (string, required): `url` or `base64`

**Returns:** Path to imported document

## Example Usage

### Analyzing a Tender Document

```python
# Using the MCP client
result = await client.call_tool("analyze_tender", {
    "file_path": "/path/to/tender.pdf"
})

# Result contains:
# - metadata: file info, page count, etc.
# - sections: document structure
# - requirements: extracted requirements
# - deadlines: important dates
# - evaluation_criteria: scoring criteria
# - compliance_items: checklist items
```

### Extracting Requirements

```python
# First parse the document
parse_result = await client.call_tool("parse_document", {
    "file_path": "/path/to/tender.pdf"
})

# Then extract requirements
requirements = await client.call_tool("extract_requirements", {
    "text": parse_result["content"]
})

# Requirements are categorized and flagged as mandatory/optional
for req in requirements["requirements"]:
    print(f"[{req['category']}] {req['description']}")
    print(f"  Mandatory: {req['is_mandatory']}")
```

### OCR Processing

```python
# Process a scanned document
ocr_result = await client.call_tool("perform_ocr", {
    "file_path": "/path/to/scanned_document.pdf"
})

print(ocr_result["ocr_text"])
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TENDER_UPLOAD_DIR` | Directory for uploaded documents | `/tmp/tender-uploads` |
| `TENDER_PROCESSED_DIR` | Directory for processed documents | `/tmp/tender-processed` |
| `TENDER_CACHE_DIR` | Cache directory | `/tmp/tender-cache` |

## Integration with tri-tender

This MCP server is designed to work seamlessly with the tri-tender document generation system:

1. **Document Upload**: Users upload tender documents via tri-tender
2. **Processing**: MCP server parses and analyzes the documents
3. **Analysis**: Extracted data feeds into the tender response generator
4. **Output**: Generated tender responses are based on comprehensive document analysis

## Requirements Categories

The MCP automatically categorizes extracted requirements:

- **Technical**: System, software, hardware specifications
- **Financial**: Pricing, payment, budget requirements
- **Legal**: Compliance, regulatory, contractual items
- **Qualification**: Experience, certifications, capacity
- **Timeline**: Deadlines, schedules, milestones
- **Documentation**: Reports, certificates, submissions
- **Personnel**: Staff, team, resource requirements
- **General**: Other requirements

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting pull requests.

## License

MIT License - see LICENSE file for details.

## Support

For issues and feature requests, please use the GitHub issue tracker.
