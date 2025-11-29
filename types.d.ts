/**
 * Tender Documents Handler MCP - TypeScript Type Definitions
 * ===========================================================
 * 
 * Type definitions for integrating the Tender Documents Handler MCP
 * with tri-tender and other TypeScript/JavaScript applications.
 */

// ============================================================================
// Enums
// ============================================================================

export enum DocumentType {
  PDF = 'pdf',
  DOCX = 'docx',
  DOC = 'doc',
  TXT = 'txt',
  HTML = 'html',
  RTF = 'rtf',
  ODT = 'odt',
  IMAGE = 'image',
  UNKNOWN = 'unknown',
}

export enum RequirementCategory {
  TECHNICAL = 'technical',
  FINANCIAL = 'financial',
  LEGAL = 'legal',
  QUALIFICATION = 'qualification',
  TIMELINE = 'timeline',
  DOCUMENTATION = 'documentation',
  PERSONNEL = 'personnel',
  GENERAL = 'general',
}

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

export interface ConvertDocumentInput {
  file_path: string;
  output_format: 'pdf' | 'txt' | 'html' | 'markdown';
  output_path?: string;
}

export interface ValidateDocumentInput {
  file_path: string;
}

export interface CompareDocumentsInput {
  file_path_1: string;
  file_path_2: string;
}

export interface SearchDocumentInput {
  file_path: string;
  query: string;
  case_sensitive?: boolean;
}

export interface GetDocumentStructureInput {
  file_path: string;
}

export interface ListUploadedDocumentsInput {
  directory?: string;
}

export interface ImportDocumentInput {
  source: string;
  filename: string;
  source_type: 'url' | 'base64';
}

// ============================================================================
// Tool Result Types
// ============================================================================

export interface ParseDocumentResult {
  file_path: string;
  document_type: DocumentType;
  text_length: number;
  content: string;
}

export interface ExtractTablesResult {
  tables: ExtractedTable[];
  count: number;
}

export interface ExtractRequirementsResult {
  requirements: TenderRequirement[];
  count: number;
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
}

export interface ConvertDocumentResult {
  success: boolean;
  input_path: string;
  output_path: string;
  format: string;
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

export interface CompareDocumentsResult {
  document_1: {
    path: string;
    word_count: number;
    sections_count: number;
    requirements_count: number;
  };
  document_2: {
    path: string;
    word_count: number;
    sections_count: number;
    requirements_count: number;
  };
  differences: {
    word_count_diff: number;
    sections_diff: number;
    requirements_diff: number;
  };
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

export interface ListUploadedDocumentsResult {
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
// MCP Tool Definitions
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
  | 'convert_document'
  | 'validate_document'
  | 'compare_documents'
  | 'search_document'
  | 'get_document_structure'
  | 'list_uploaded_documents'
  | 'import_document';

// ============================================================================
// Utility Types
// ============================================================================

export interface MCPError {
  error: string;
}

export type MCPResult<T> = T | MCPError;

export function isError(result: MCPResult<any>): result is MCPError {
  return 'error' in result;
}

// ============================================================================
// Integration Helper
// ============================================================================

/**
 * Helper class for working with the Tender Documents MCP from tri-tender
 */
export interface TenderDocsMCPClient {
  parseDocument(input: ParseDocumentInput): Promise<MCPResult<ParseDocumentResult>>;
  analyzeTender(input: AnalyzeTenderInput): Promise<MCPResult<AnalysisResult>>;
  extractMetadata(input: ExtractMetadataInput): Promise<MCPResult<DocumentMetadata>>;
  extractTables(input: ExtractTablesInput): Promise<MCPResult<ExtractTablesResult>>;
  extractRequirements(input: ExtractRequirementsInput): Promise<MCPResult<ExtractRequirementsResult>>;
  extractSections(input: ExtractSectionsInput): Promise<MCPResult<ExtractSectionsResult>>;
  extractDeadlines(input: ExtractDeadlinesInput): Promise<MCPResult<ExtractDeadlinesResult>>;
  performOCR(input: PerformOCRInput): Promise<MCPResult<PerformOCRResult>>;
  convertDocument(input: ConvertDocumentInput): Promise<MCPResult<ConvertDocumentResult>>;
  validateDocument(input: ValidateDocumentInput): Promise<MCPResult<ValidateDocumentResult>>;
  compareDocuments(input: CompareDocumentsInput): Promise<MCPResult<CompareDocumentsResult>>;
  searchDocument(input: SearchDocumentInput): Promise<MCPResult<SearchDocumentResult>>;
  getDocumentStructure(input: GetDocumentStructureInput): Promise<MCPResult<DocumentStructure>>;
  listUploadedDocuments(input?: ListUploadedDocumentsInput): Promise<MCPResult<ListUploadedDocumentsResult>>;
  importDocument(input: ImportDocumentInput): Promise<MCPResult<ImportDocumentResult>>;
}
