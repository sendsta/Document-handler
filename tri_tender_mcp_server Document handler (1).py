"""
Tri‑Tender MCP Server
======================

This script provides a simple command‑line interface around the core
``tri_tender_mcp`` module to implement the Tri‑Tender Tender Document
Intelligence Workflow.  When invoked with one or more file paths, it
automatically detects the type of each document, extracts critical
metadata, generates a structured executive summary and presents the
official Tri‑Tender action menu.

The server does not require any external services or network access.  It
uses the local filesystem to read uploaded documents.  This makes it
suitable for integration into offline environments or as the core of a
larger MCP server when additional tooling becomes available.

Usage:

    python tri_tender_mcp_server.py <file1> [<file2> ...]

Upon completion, the script prints a JSON object to stdout containing
the classification of each file, the merged metadata extracted from all
files and a combined summary.  The summary always ends with the
standard Tri‑Tender action menu so that downstream agents or users can
select the next step.

You can import ``run_workflow`` from this module in your own code to
programmatically invoke the automatic tender document handler without
using the command‑line interface.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, Any, List

# Import the core processing functions from tri_tender_mcp
try:
    from tri_tender_mcp import process_files, process_file  # type: ignore
except ImportError:
    # If tri_tender_mcp is not in the Python path, attempt to load it from
    # the same directory as this script.
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).parent / "tri_tender_mcp.py"
    spec = importlib.util.spec_from_file_location("tri_tender_mcp", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to locate tri_tender_mcp module")
    tri_tender_mcp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tri_tender_mcp)  # type: ignore
    process_files = tri_tender_mcp.process_files  # type: ignore
    process_file = tri_tender_mcp.process_file  # type: ignore


def run_workflow(paths: List[str]) -> Dict[str, Any]:
    """Run the Tri‑Tender workflow on one or more files and return results.

    Given a list of file paths, this function processes each file using
    the core ``tri_tender_mcp`` logic, merges metadata and returns a
    dictionary with the following keys:

    ``classifications``: a mapping from classification name to list of file
        paths that were given that classification.

    ``metadata``: a dictionary representation of the merged
        ``TenderMetadata`` object extracted from all files.

    ``summary``: a combined summary string containing one summary per
        processed file, separated by blank lines.

    If any file does not exist, a KeyError is raised.
    """
    # Verify files exist
    for p in paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"File not found: {p}")
    return process_files(paths)


def main(argv: List[str]) -> None:
    """CLI entry point.

    Parses command‑line arguments and runs the workflow.  The result
    dictionary is printed as pretty‑formatted JSON on stdout.  If no
    arguments are given, prints a usage message and exits with code 1.
    """
    if len(argv) < 2:
        cmd = os.path.basename(argv[0]) if argv else "tri_tender_mcp_server.py"
        print(f"Usage: {cmd} <file1> [<file2> ...]", file=sys.stderr)
        print("Automatically process tender documents and output metadata and summary.", file=sys.stderr)
        sys.exit(1)
    files = argv[1:]
    try:
        result = run_workflow(files)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    # Pretty print JSON result
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(sys.argv)