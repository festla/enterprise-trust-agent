from enum import Enum


class Exchange(str, Enum):
    """证券交易所。"""

    SZSE = "SZSE"
    SSE = "SSE"



class RecordStatus(str, Enum):
    """业务记录状态。"""

    DRAFT = "draft"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"


class ReportType(str, Enum):
    """报告类型。"""

    ANNUAL_REPORT = "annual_report"
    SEMIANNUAL_REPORT = "semiannual_report"
    QUARTERLY_REPORT = "quarterly_report"
    OTHER = "other"


class DocumentQualityGrade(str, Enum):
    """文档质量等级。"""

    A = "A"
    B = "B"
    C = "C"


class Severity(str, Enum):
    """风险或问题严重程度。"""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ValidationStatus(str, Enum):
    """数据核验状态。"""

    PENDING = "pending"
    VERIFIED = "verified"
    DISPUTED = "disputed"
    REJECTED = "rejected"


class PageMappingRuleType(str, Enum):
    """印刷页码与 PDF 页码的映射类型。"""

    IDENTITY = "identity"
    OFFSET = "offset"
    CUSTOM = "custom"


class StatementType(str, Enum):
    """财务报表或披露章节类型。"""

    BALANCE_SHEET = "balance_sheet"
    INCOME_STATEMENT = "income_statement"
    CASH_FLOW_STATEMENT = "cash_flow_statement"
    STATEMENT_OF_CHANGES_IN_EQUITY = "statement_of_changes_in_equity"
    FINANCIAL_SUMMARY = "financial_summary"
    NOTE = "note"
    MANAGEMENT_DISCUSSION = "management_discussion"
    IMPORTANT_EVENTS = "important_events"
    OTHER = "other"


class StatementScope(str, Enum):
    """报表或指标口径。"""

    CONSOLIDATED = "consolidated"
    PARENT_COMPANY = "parent_company"
    SEGMENT = "segment"
    GROUP = "group"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class PeriodType(str, Enum):
    """指标所属期间类型。"""

    INSTANT = "instant"
    DURATION = "duration"


class MetricOrigin(str, Enum):
    """指标的产生方式。"""

    REPORTED = "reported"
    DERIVED = "derived"


class UnitCode(str, Enum):
    """财务数据单位。"""

    CNY = "CNY"
    CNY_THOUSAND = "CNY_thousand"
    CNY_TEN_THOUSAND = "CNY_ten_thousand"
    CNY_MILLION = "CNY_million"
    CNY_HUNDRED_MILLION = "CNY_hundred_million"
    PERCENT = "percent"
    PERCENTAGE_POINT = "percentage_point"
    RATIO = "ratio"
    CNY_PER_SHARE = "CNY_per_share"
    COUNT = "count"
    TEXT = "text"


class AliasMatchType(str, Enum):
    """指标别名匹配方式。"""

    EXACT = "exact"
    NORMALIZED = "normalized"
    REGEX = "regex"
    SEMANTIC = "semantic"


class MetricValueType(str, Enum):
    """指标值的数据类型。"""

    DECIMAL = "decimal"
    INTEGER = "integer"
    TEXT = "text"


class EvidenceType(str, Enum):
    """来源证据类型。"""

    FINANCIAL_STATEMENT_CELL = "financial_statement_cell"
    FINANCIAL_SUMMARY_TABLE = "financial_summary_table"
    MANAGEMENT_STATEMENT = "management_statement"
    FINANCIAL_NOTE = "financial_note"
    RISK_DISCLOSURE = "risk_disclosure"
    TABLE = "table"
    PARAGRAPH = "paragraph"
    CALCULATION_INPUT = "calculation_input"
    OTHER = "other"


class AttributionType(str, Enum):
    """结论或证据的归属类型。"""

    REPORT_DISCLOSURE = "report_disclosure"
    MANAGEMENT_STATEMENT = "management_statement"
    SYSTEM_CALCULATION = "system_calculation"
    DERIVED_INFERENCE = "derived_inference"


class RestatementStatus(str, Enum):
    """比较数据的重列状态。"""

    NOT_APPLICABLE = "not_applicable"
    NOT_RESTATED = "not_restated"
    RESTATED = "restated"
    UNKNOWN = "unknown"


class EvidenceSupportType(str, Enum):
    """证据对财务事实的支持类型。"""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    CONTEXT = "context"


class PageCountStatus(str, Enum):
    """PDF 实际页数与 Registry 预期页数的比较结果。"""

    MATCHED = "matched"
    MISMATCHED = "mismatched"


class DocumentValidationStatus(str, Enum):
    """实际 PDF 文件能否进入后续页面解析管线。"""

    VALID = "valid"
    BLOCKED = "blocked"


class PageMappingStatus(str, Enum):
    """单个 PDF 页面的印刷页码映射状态。"""

    MAPPED = "mapped"
    UNMAPPED = "unmapped"


class PageContentType(str, Enum):
    """页面中可供后续处理的主要内容类型。"""

    TEXT = "text"
    EMPTY = "empty"
    SCANNED = "scanned"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class PageParseStatus(str, Enum):
    """单个页面的解析执行状态。"""

    SUCCESS = "success"
    PARSE_ERROR = "parse_error"


class ChunkStrategy(str, Enum):
    """页面文本切分策略。"""

    FIXED_LENGTH = "fixed_length"
    PARAGRAPH = "paragraph"
    SECTION_PARAGRAPH = "section_paragraph"