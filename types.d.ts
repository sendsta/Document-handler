/**
 * Tender Documents Handler MCP - TypeScript Type Definitions
 * ===========================================================
 * 
 * Type definitions for integrating the Tender Documents Handler MCP
 * with tri-tender and other TypeScript/JavaScript applications.
 * 
 * Compatible with FastMCP Cloud deployment.
 */

// ============================================================================
// Enums
// ============================================================================

export type DocumentType = 
  | 'pdf' 
  | 'docx' 
  | 'doc' 
  | 'txt' 
  | 'html' 
  | 'rtf' 
  | 'odt' 
  | 'image' 
  | 'unknown';

export type RequirementCategory = 
  | 'technical'
  | 'financial'
  | 'legal'
  | 'qualification'
  | 'timeline'
  | 'documentation'
  | 'personnel'
  | 'general';

// ============================================================================
// Core Interfaces
// ============================================================================

export interface DocumentMetadata {
  file_name: string;
  file_path: string;
  file_size: number;
  file_type: DocumentType;
  page_count?: number;
  title?: string;
  author?: string;
  created_date?: string;
  modified_date?: string;
  word_count?: number;
  checksum?: string;
}

export interface ExtractedSection {
  title: string;
  content: string;
  page_number?: number;
  section_number?: string;
  level: number;
}

export interface TenderRequirement {
  requirement_id: string;
  description: string;
  category: RequirementCategory;
  is_mandatory: boolean;
  page_number?: number;
  source_text?: string;
}

export interface Deadline {
  date: string;
  context: string;
  type: string;
}

export interface Contact {
  type: 'email' | 'phone';
  value: string;
}

export interface EvaluationCriteria {
  description: string;
  source: string;
  weight?: number;
}

export interface ComplianceItem {
  item: string;
  completed: boolean;
}

export interface ExtractedTable {
  page: number;
  table_index: number;
  headers: string[];
  rows: string[][];
  row_count: number;
}

// ============================================================================
// Analysis Result
// ============================================================================

export interface AnalysisResult {
  document_id: string;
  metadata: DocumentMetadata;
  sections: ExtractedSection[];
  requirements: TenderRequirement[];
  tables: ExtractedTable[];
  deadlines: Deadline[];
  key_contacts: Contact[];
  evaluation_criteria: EvaluationCriteria[];
  compliance_items: ComplianceItem[];
  full_text: string;
  summary: string;
}

// ============================================================================
// Tool Input Types
// ============================================================================

export interface ParseDocumentInput {
  file_path: string;
  use_ocr?: boolean;
}

export interface AnalyzeTenderInput {
  file_path: string;
}

export interface ExtractMetadataInput {
  file_path: string;
}

export interface ExtractTablesInput {
  file_path: string;
}

export interface ExtractRequirementsInput {
  text: string;
}

export interface ExtractSectionsInput {
  text: string;
}

export interface ExtractDeadlinesInput {
  text: string;
}

export interface PerformOCRInput {
  file_path: string;
}

export interface ValidateDocumentInput {
  file_path: string;
}

export interface SearchDocumentInput {
  file_path: string;
  query: string;
  case_sensitive?: boolean;
}

export interface GetDocumentStructureInput {
  file_path: string;
}

export interface ListDocumentsInput {
  directory?: string;
}

export interface ImportDocumentInput {
  source: string;
  filename: string;
  source_type: 'base64';
}

// ============================================================================
// Tool Result Types
// ============================================================================

export interface ParseDocumentResult {
  file_path: string;
  document_type: DocumentType;
  text_length: number;
  word_count: number;
  content: string;
}

export interface ExtractTablesResult {
  tables: ExtractedTable[];
  count: number;
}

export interface ExtractRequirementsResult {
  requirements: TenderRequirement[];
  count: number;
  mandatory_count: number;
}

export interface ExtractSectionsResult {
  sections: ExtractedSection[];
  count: number;
}

export interface ExtractDeadlinesResult {
  deadlines: Deadline[];
  count: number;
}

export interface PerformOCRResult {
  file_path: string;
  ocr_text: string;
  text_length: number;
  word_count: number;
}

export interface ValidateDocumentResult {
  file_path: string;
  exists: boolean;
  readable: boolean;
  has_content: boolean;
  valid: boolean;
  document_type?: DocumentType;
  word_count?: number;
  issues: string[];
}

export interface SearchMatch {
  position: number;
  context: string;
}

export interface SearchDocumentResult {
  query: string;
  matches_found: number;
  matches: SearchMatch[];
}

export interface DocumentStructure {
  document: string;
  sections: Array<{
    number: string;
    title: string;
    level: number;
    content_preview: string;
  }>;
  table_of_contents: Array<{
    level: number;
    title: string;
  }>;
}

export interface UploadedDocument {
  name: string;
  path: string;
  type: DocumentType;
  size: number;
  modified: string;
}

export interface ListDocumentsResult {
  directory: string;
  documents: UploadedDocument[];
  count: number;
}

export interface ImportDocumentResult {
  success: boolean;
  file_path: string;
  file_size: number;
}

// ============================================================================
// Resource Types
// ============================================================================

export interface SupportedFormats {
  documents: string[];
  images: string[];
  ocr_supported: boolean;
}

export interface RequirementCategoryInfo {
  id: RequirementCategory;
  description: string;
}

// ============================================================================
// MCP Tool Names
// ============================================================================

export type TenderDocsTool =
  | 'parse_document'
  | 'analyze_tender'
  | 'extract_metadata'
  | 'extract_tables'
  | 'extract_requirements'
  | 'extract_sections'
  | 'extract_deadlines'
  | 'perform_ocr'
  | 'validate_document'
  | 'search_document'
  | 'get_document_structure'
  | 'list_documents'
  | 'import_document';

// ============================================================================
// MCP Resource URIs
// ============================================================================

export type TenderDocsResource =
  | 'tender://config/version'
  | 'tender://config/supported-formats'
  | 'tender://config/requirement-categories'
  | `tender://uploads/${string}`;

// ============================================================================
// MCP Prompt Names
// ============================================================================

export type TenderDocsPrompt =
  | 'analyze_tender_prompt'
  | 'compare_requirements_prompt'
  | 'extract_compliance_checklist_prompt';

// ============================================================================
// Utility Types
// ============================================================================

export interface MCPError {
  error: string;
}

export type MCPResult<T> = T | MCPError;

export function isError(result: MCPResult<any>): result is MCPError {
  return result && typeof result === 'object' && 'error' in result;
}

// ============================================================================
// FastMCP Cloud Configuration
// ============================================================================

export interface FastMCPCloudConfig {
  url: string;
  headers?: {
    Authorization?: string;
    [key: string]: string | undefined;
  };
}

export interface MCPClientConfig {
  mcpServers: {
    'tender-docs': FastMCPCloudConfig;
  };
}

/**
 * Example configuration for Claude Desktop or Cursor
 */
export const exampleConfig: MCPClientConfig = {
  mcpServers: {
    'tender-docs': {
      url: 'https://your-project-name.fastmcp.app/mcp'
    }
  }
};

/**
 * Example configuration with authentication
 */
export const exampleAuthConfig: MCPClientConfig = {
  mcpServers: {
    'tender-docs': {
      url: 'https://your-project-name.fastmcp.app/mcp',
      headers: {
        Authorization: 'Bearer YOUR_TOKEN'
      }
    }
  }
};
