"""
Tests for Tender Documents Handler FastMCP Server
"""

import pytest
import tempfile
import os
from pathlib import Path

# Import from server
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from server import (
    mcp,
    detect_document_type,
    extract_sections,
    extract_requirements_from_text,
    extract_deadlines_from_text,
    extract_contacts_from_text,
    categorize_requirement,
    DocumentType,
)


class TestDocumentTypeDetection:
    """Tests for document type detection."""
    
    def test_detect_pdf(self):
        assert detect_document_type(Path("test.pdf")) == DocumentType.PDF
    
    def test_detect_docx(self):
        assert detect_document_type(Path("test.docx")) == DocumentType.DOCX
    
    def test_detect_doc(self):
        assert detect_document_type(Path("test.doc")) == DocumentType.DOC
    
    def test_detect_txt(self):
        assert detect_document_type(Path("test.txt")) == DocumentType.TXT
    
    def test_detect_html(self):
        assert detect_document_type(Path("test.html")) == DocumentType.HTML
        assert detect_document_type(Path("test.htm")) == DocumentType.HTML
    
    def test_detect_image(self):
        assert detect_document_type(Path("test.png")) == DocumentType.IMAGE
        assert detect_document_type(Path("test.jpg")) == DocumentType.IMAGE
        assert detect_document_type(Path("test.jpeg")) == DocumentType.IMAGE
    
    def test_detect_unknown(self):
        assert detect_document_type(Path("test.xyz")) == DocumentType.UNKNOWN


class TestSectionExtraction:
    """Tests for section extraction."""
    
    def test_numbered_sections(self):
        text = """
1. INTRODUCTION
This is the introduction section.

2. SCOPE OF WORK
This section describes the scope.

2.1 Technical Requirements
Technical details here.
"""
        sections = extract_sections(text)
        assert len(sections) >= 2
    
    def test_caps_headers(self):
        text = """
EXECUTIVE SUMMARY
This is the executive summary.

BACKGROUND
This provides background information.
"""
        sections = extract_sections(text)
        assert len(sections) >= 1


class TestRequirementExtraction:
    """Tests for requirement extraction."""
    
    def test_must_requirements(self):
        text = "The contractor must provide weekly reports. The system must be available 24/7."
        requirements = extract_requirements_from_text(text)
        assert len(requirements) >= 1
    
    def test_shall_requirements(self):
        text = "The vendor shall deliver within 30 days. The solution shall meet ISO standards."
        requirements = extract_requirements_from_text(text)
        assert len(requirements) >= 1
    
    def test_mandatory_detection(self):
        text = "This is a mandatory requirement: all staff must be certified."
        requirements = extract_requirements_from_text(text)
        assert any(r.is_mandatory for r in requirements)


class TestRequirementCategorization:
    """Tests for requirement categorization."""
    
    def test_technical_category(self):
        assert categorize_requirement("Must implement the software system") == "technical"
    
    def test_financial_category(self):
        assert categorize_requirement("Payment must be within 30 days") == "financial"
    
    def test_legal_category(self):
        assert categorize_requirement("Must comply with all regulations") == "legal"
    
    def test_qualification_category(self):
        assert categorize_requirement("Must have 5 years experience") == "qualification"
    
    def test_timeline_category(self):
        assert categorize_requirement("Delivery deadline is firm") == "timeline"


class TestDeadlineExtraction:
    """Tests for deadline extraction."""
    
    def test_date_formats(self):
        text = """
        Submission deadline: 15/03/2024
        The closing date is March 30, 2024.
        Due by January 15, 2024.
        """
        deadlines = extract_deadlines_from_text(text)
        assert len(deadlines) >= 1
    
    def test_deadline_context(self):
        text = "The submission deadline is 01/04/2024. Please ensure all documents are received."
        deadlines = extract_deadlines_from_text(text)
        assert len(deadlines) >= 1


class TestContactExtraction:
    """Tests for contact extraction."""
    
    def test_email_extraction(self):
        text = "Contact us at procurement@example.com or support@tender.gov"
        contacts = extract_contacts_from_text(text)
        emails = [c for c in contacts if c["type"] == "email"]
        assert len(emails) >= 2
    
    def test_phone_extraction(self):
        text = "Call us at +1-555-123-4567 or 0800 123 456"
        contacts = extract_contacts_from_text(text)
        phones = [c for c in contacts if c["type"] == "phone"]
        assert len(phones) >= 1


class TestFastMCPServer:
    """Tests for FastMCP server configuration."""
    
    def test_server_name(self):
        assert mcp.name == "Tender Documents Handler"
    
    def test_server_has_tools(self):
        # Check that tools are registered
        assert hasattr(mcp, '_tool_manager')


class TestIntegration:
    """Integration tests with sample files."""
    
    def test_process_text_file(self):
        """Test processing a simple text file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("""
TENDER DOCUMENT

1. INTRODUCTION
This tender seeks proposals for IT services.

2. REQUIREMENTS
The contractor must have ISO certification.
The vendor shall provide 24/7 support.

Submission deadline: December 31, 2024

Contact: tender@example.com
            """)
            f.flush()
            
            try:
                file_path = Path(f.name)
                
                # Read the file
                with open(file_path, 'r') as rf:
                    text = rf.read()
                
                # Test extraction functions
                sections = extract_sections(text)
                requirements = extract_requirements_from_text(text)
                deadlines = extract_deadlines_from_text(text)
                contacts = extract_contacts_from_text(text)
                
                assert len(sections) >= 1
                assert len(requirements) >= 1
                assert len(contacts) >= 1
                
            finally:
                os.unlink(f.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
