"""
Tender Documents Handler MCP
============================

A Model Context Protocol (MCP) server for parsing, analyzing, importing,
OCR processing, and managing tender documents for tri-tender.

Usage:
    # Run the MCP server
    python -m tender_docs_mcp
    
    # Or import components
    from tender_docs_mcp import (
        analyze_tender_document,
        extract_text_from_pdf,
        extract_requirements,
        extract_sections,
    )
"""

from .tender_docs_mcp import (
    # Main analysis function
    analyze_tender_document,
    
    # Document parsing
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_html,
    perform_ocr,
    
    # Metadata extraction
    extract_metadata,
    detect_document_type,
    calculate_checksum,
    
    # Analysis functions
    extract_sections,
    extract_requirements,
    extract_deadlines,
    extract_contacts,
    extract_evaluation_criteria,
    extract_compliance_items,
    extract_tables_from_pdf,
    
    # Data classes
    DocumentMetadata,
    ExtractedSection,
    TenderRequirement,
    AnalysisResult,
    DocumentType,
    
    # Server
    server,
    main,
)

__version__ = "1.0.0"
__author__ = "Tri-Tender Team"
__all__ = [
    # Main functions
    "analyze_tender_document",
    "main",
    "server",
    
    # Parsing
    "extract_text_from_pdf",
    "extract_text_from_docx",
    "extract_text_from_html",
    "perform_ocr",
    
    # Metadata
    "extract_metadata",
    "detect_document_type",
    "calculate_checksum",
    
    # Analysis
    "extract_sections",
    "extract_requirements",
    "extract_deadlines",
    "extract_contacts",
    "extract_evaluation_criteria",
    "extract_compliance_items",
    "extract_tables_from_pdf",
    
    # Data classes
    "DocumentMetadata",
    "ExtractedSection",
    "TenderRequirement",
    "AnalysisResult",
    "DocumentType",
]
