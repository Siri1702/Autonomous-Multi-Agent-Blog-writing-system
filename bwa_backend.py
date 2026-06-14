from __future__ import annotations

import operator
import os
import re
import json
import logging
import uuid
from datetime import date, timedelta, datetime
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated, Any
from enum import Enum

from pydantic import BaseModel, Field, ValidationError

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# PHASE 1: Enhanced State Management - Enums & Constants
# ============================================================

class BlogStatus(str, Enum):
    """Status of a blog generation run."""
    DRAFT = "draft"
    PLANNING = "planning"
    RESEARCHING = "researching"
    PLAN_REVIEW = "plan_review"  # PHASE 1.3: Awaiting user approval
    WRITING = "writing"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    FAILED = "failed"


class OperationType(str, Enum):
    """Types of operations in the audit trail."""
    ROUTER_DECISION = "router_decision"
    RESEARCH_COMPLETE = "research_complete"
    PLAN_CREATED = "plan_created"
    PLAN_APPROVED = "plan_approved"
    PLAN_REJECTED = "plan_rejected"
    WORKER_SECTION = "worker_section"
    MERGE_COMPLETE = "merge_complete"
    FINAL_OUTPUT = "final_output"
    ERROR = "error"

# ============================================================
# Blog Writer (Router → (Research?) → Orchestrator → Workers → Reducer)
#
# ============================================================


# -----------------------------
# 1) Schemas
# -----------------------------
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should do/understand.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target words (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False

    # PHASE 2: Technical Depth Tiers
    depth_level: Literal["beginner", "intermediate", "expert"] = "intermediate"
    code_complexity: int = Field(default=3, ge=1, le=5)
    requires_prerequisites: List[str] = Field(default_factory=list)


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)

    # PHASE 2: Technical Depth Tiers
    depth_level: Literal["beginner", "intermediate", "expert"] = "intermediate"
    target_audience_description: str = Field(default="", description="Description of target audience expertise")

    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # ISO "YYYY-MM-DD" preferred
    snippet: Optional[str] = None
    source: Optional[str] = None

    # PHASE 2: Citation Quality System
    source_quality: Literal["high", "medium", "low"] = "medium"
    source_type: Optional[Literal["academic", "official_docs", "community", "news", "blog"]] = None


# ============================================================
# PHASE 2: Multi-Source Research - Schemas
# ============================================================

class SourceType(str, Enum):
    """Types of research sources."""
    TAVILY = "tavily"
    ARXIV = "arxiv"
    GITHUB = "github"
    DOCS = "docs"


class ClaimVerification(BaseModel):
    """Verification status for a factual claim."""
    claim: str
    verified: bool
    evidence_url: Optional[str] = None
    verification_notes: Optional[str] = None


class ResearchResult(BaseModel):
    """Result from multi-source research."""
    source: SourceType
    items: List[EvidenceItem]
    query: str
    success: bool = True
    error_message: Optional[str] = None


# ============================================================
# PHASE 1: Enhanced State Management - Schemas
# ============================================================

class BlogMetadata(BaseModel):
    """Metadata for a blog generation run."""
    blog_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    status: BlogStatus = BlogStatus.DRAFT
    version: int = 1
    generation_time_seconds: Optional[float] = None

    # Content stats
    word_count: int = 0
    reading_time_minutes: float = 0
    sections_count: int = 0

    # Quality metrics (to be filled later)
    seo_score: Optional[float] = None
    readability_score: Optional[float] = None


class AuditEntry(BaseModel):
    """Single entry in the audit trail."""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    operation: OperationType
    node: str
    details: str
    success: bool = True
    error_message: Optional[str] = None


class BlogVersion(BaseModel):
    """Version tracking for blog drafts."""
    version: int
    created_at: str
    content: str
    plan_snapshot: Optional[str] = None  # JSON string of plan at this version
    metadata_snapshot: Optional[BlogMetadata] = None


# ============================================================
# PHASE 2: Technical Depth Tiers - Schemas
# ============================================================

class TechnicalDepth(BaseModel):
    """Technical depth configuration for a task."""
    level: Literal["beginner", "intermediate", "expert"] = "intermediate"
    code_complexity: int = Field(default=3, ge=1, le=5, description="1=minimal, 5=advanced")
    requires_prerequisites: List[str] = Field(default_factory=list, description="Prerequisites reader should know")
    explanation_style: Literal["analogy", "practical", "deep_dive"] = "practical"


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(5)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


# ============================================================
# PHASE 1.3: Outline Approval Workflow - Schemas
# ============================================================

class PlanRevision(BaseModel):
    """User feedback for plan revision."""
    approved: bool = Field(..., description="Whether the plan is approved")
    revision_notes: Optional[str] = Field(default=None, description="Notes for revisions if not approved")
    modified_tasks: Optional[List[Task]] = Field(default=None, description="Modified task list if user made changes")


class State(TypedDict):
    topic: str

    # PHASE 1: Metadata & Versioning
    metadata: BlogMetadata
    audit_trail: Annotated[List[AuditEntry], operator.add]
    versions: List[BlogVersion]

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # PHASE 1.3: Plan Review
    plan_approved: bool
    plan_revision_notes: Optional[str]
    awaiting_plan_approval: bool

    # PHASE 2: Fact-Checking
    claim_verifications: Optional[List[ClaimVerification]]

    # PHASE 3: Technical Content
    diagrams: Optional[List[dict]]
    comparison_matrices: Optional[List[dict]]

    # recency
    as_of: str
    recency_days: int

    # PHASE 2: Technical Depth Tiers
    depth_level: str

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)

    # reducer
    merged_md: str
    md_with_placeholders: str

    final: str


# -----------------------------
# 2) LLM Configuration
# -----------------------------
# Use the model specified in CLAUDE.md
llm = ChatOllama(model="minimax-m2.5:cloud")


# -----------------------------
# 3) Helper Functions for Strict JSON Output
# -----------------------------
def validate_json_output(output: Any, schema_type: str) -> Optional[dict]:
    """
    Validate and parse JSON output from LLM.
    Returns parsed dict or None if validation fails.
    """
    try:
        if hasattr(output, 'model_dump'):
            return output.model_dump()
        elif isinstance(output, dict):
            return output
        elif isinstance(output, str):
            # Try to extract JSON from string
            json_match = re.search(r'\{.*\}', output, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        return None
    except (ValidationError, json.JSONDecodeError, AttributeError) as e:
        logger.warning(f"JSON validation failed for {schema_type}: {e}")
        return None


def create_strict_prompt(base_prompt: str, schema_example: str, schema_name: str) -> str:
    """
    Create a strict prompt that enforces exact JSON output format.
    """
    return f"""{base_prompt}

STRICT OUTPUT REQUIREMENTS:
- You MUST output ONLY valid JSON that matches the {schema_name} schema exactly
- Do NOT include any explanatory text, markdown formatting, or code blocks
- Your response must be parseable by json.loads() without any preprocessing
- Include all required fields as defined in the schema

SCHEMA EXAMPLE:
```json
{schema_example}
```

OUTPUT MUST BE ONLY THE JSON OBJECT. NOTHING ELSE."""


# ============================================================
# PHASE 1: Enhanced State Management - Helper Functions
# ============================================================

def create_initial_state(topic: str, as_of: str = None, depth_level: str = "intermediate") -> State:
    """
    Create initial state with metadata, versioning, and audit trail.

    Args:
        topic: The blog topic
        as_of: The date for research recency (default: today)
        depth_level: Technical depth level (beginner, intermediate, expert)
    """
    if as_of is None:
        as_of = date.today().isoformat()

    metadata = BlogMetadata(
        blog_id=str(uuid.uuid4()),
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
        status=BlogStatus.DRAFT,
        version=1
    )

    return {
        "topic": topic,
        "metadata": metadata,
        "audit_trail": [],
        "versions": [],
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "plan_approved": False,  # PHASE 1.3
        "plan_revision_notes": None,
        "awaiting_plan_approval": False,
        "claim_verifications": [],  # PHASE 2: Fact-checking
        "diagrams": [],  # PHASE 3: Diagrams
        "comparison_matrices": [],  # PHASE 3: Comparison matrices
        "as_of": as_of,
        "recency_days": 365,
        "depth_level": depth_level,  # PHASE 2: Store depth level in state
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "final": "",
    }


def add_audit_entry(
    state: State,
    operation: OperationType,
    node: str,
    details: str,
    success: bool = True,
    error_message: Optional[str] = None
) -> dict:
    """
    Add an entry to the audit trail.
    """
    entry = AuditEntry(
        timestamp=datetime.now().isoformat(),
        operation=operation,
        node=node,
        details=details,
        success=success,
        error_message=error_message
    )
    return {"audit_trail": [entry]}


def update_metadata(state: State, **updates) -> dict:
    """
    Update metadata fields and track the update time.
    """
    current_metadata = state.get("metadata", BlogMetadata())
    metadata_dict = current_metadata.model_dump()

    for key, value in updates.items():
        if key in metadata_dict:
            metadata_dict[key] = value

    metadata_dict["updated_at"] = datetime.now().isoformat()

    return {"metadata": BlogMetadata(**metadata_dict)}


def save_version(state: State, content: str, plan_snapshot: Optional[Plan] = None) -> dict:
    """
    Save current state as a new version.
    """
    current_version = len(state.get("versions", [])) + 1

    version = BlogVersion(
        version=current_version,
        created_at=datetime.now().isoformat(),
        content=content,
        plan_snapshot=plan_snapshot.model_dump_json() if plan_snapshot else None,
        metadata_snapshot=state.get("metadata")
    )

    # Update metadata version
    metadata = state.get("metadata")
    if metadata:
        metadata.version = current_version

    return {"versions": [version]}


# -----------------------------
# 4) Router - with strict JSON enforcement
# -----------------------------
ROUTER_SCHEMA_EXAMPLE = """{
  "needs_research": true,
  "mode": "open_book",
  "reason": "The topic requires current information about latest developments",
  "queries": ["query 1", "query 2", "query 3"],
  "max_results_per_query": 5
}"""

ROUTER_SYSTEM = create_strict_prompt(
    """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen concepts.
- hybrid (needs_research=true): evergreen + needs up-to-date examples/tools/models.
- open_book (needs_research=true): volatile weekly/news/"latest"/pricing/policy.

If needs_research=true:
- Output 3–10 high-signal, scoped queries.
- For open_book weekly roundup, include queries reflecting last 7 days.""",
    ROUTER_SCHEMA_EXAMPLE,
    "RouterDecision"
)


def router_node(state: State) -> dict:
    """Router node that decides the mode and research needs."""
    max_retries = 3
    last_error = None

    # Update status to planning
    status_update = update_metadata(state, status=BlogStatus.PLANNING)

    for attempt in range(max_retries):
        try:
            decider = llm.with_structured_output(RouterDecision)
            decision = decider.invoke(
                [
                    SystemMessage(content=ROUTER_SYSTEM),
                    HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
                ]
            )

            # Validate output
            if decision is None:
                raise ValueError("Router returned None")

            validated = validate_json_output(decision, "RouterDecision")
            if validated is None:
                raise ValueError("Router output validation failed")

            if decision.mode == "open_book":
                recency_days = 7
            elif decision.mode == "hybrid":
                recency_days = 45
            else:
                recency_days = 3650

            # Add audit entry for successful routing
            audit_entry = add_audit_entry(
                state,
                operation=OperationType.ROUTER_DECISION,
                node="router",
                details=f"Mode: {decision.mode}, Research needed: {decision.needs_research}, Queries: {len(decision.queries)}",
                success=True
            )

            return {
                "needs_research": decision.needs_research,
                "mode": decision.mode,
                "queries": decision.queries,
                "recency_days": recency_days,
                **status_update,
                **audit_entry,
            }

        except (ValidationError, ValueError, AttributeError) as e:
            last_error = e
            logger.warning(f"Router attempt {attempt + 1} failed: {e}")
            continue

    # Fallback to closed_book if all retries fail
    logger.error(f"All router retries failed: {last_error}. Falling back to closed_book.")

    # Add audit entry for failed routing
    audit_entry = add_audit_entry(
        state,
        operation=OperationType.ROUTER_DECISION,
        node="router",
        details=f"Router failed after {max_retries} retries, falling back to closed_book",
        success=False,
        error_message=str(last_error)
    )

    return {
        "needs_research": False,
        "mode": "closed_book",
        "queries": [],
        "recency_days": 3650,
        **audit_entry,
    }


def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"


# -----------------------------
# 5) Research (Tavily) - with strict JSON enforcement
# -----------------------------
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore
        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})
        out: List[dict] = []
        for r in results or []:
            out.append(
                {
                    "title": r.get("title") or "",
                    "url": r.get("url") or "",
                    "snippet": r.get("content") or r.get("snippet") or "",
                    "published_date": r.get("published_date") or r.get("published_at"),
                    "source": r.get("source"),
                }
            )
        return out
    except Exception:
        return []


def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


# ============================================================
# PHASE 2: Multi-Source Research Functions
# ============================================================

def _arxiv_search(query: str, max_results: int = 5) -> List[dict]:
    """
    Search arXiv for academic papers.
    """
    try:
        from langchain_community.tools import ArxivQueryRun
        tool = ArxivQueryRun(api_description="Searches arXiv for academic papers")
        # Arxiv tool doesn't accept max_results, so we'll take top results
        results = tool.invoke(query)
        if not results:
            return []

        # Parse arXiv results
        out = []
        if isinstance(results, str):
            # Parse the text result
            for line in results.split("\n"):
                if "arXiv:" in line:
                    arxiv_id = line.split("arXiv:")[-1].strip()
                    out.append({
                        "title": f"arXiv Paper: {arxiv_id}",
                        "url": f"https://arxiv.org/abs/{arxiv_id}",
                        "snippet": line[:200],
                        "source": "arXiv",
                        "source_type": "academic",
                    })
        return out[:max_results]
    except Exception:
        return []


def _github_search(query: str, max_results: int = 5) -> List[dict]:
    """
    Search GitHub for repositories (via Tavily with GitHub focus).
    """
    # Use Tavily but filter for GitHub results
    tavily_results = _tavily_search(f"{query} github", max_results)
    github_results = []
    for r in tavily_results:
        if "github.com" in r.get("url", ""):
            r["source_type"] = "community"
            github_results.append(r)
    return github_results


def _docs_search(query: str, max_results: int = 5) -> List[dict]:
    """
    Search for official documentation.
    """
    # Search for documentation sites
    doc_queries = [
        f"{query} documentation site:docs.python.org OR site:docs.microsoft.com OR site:developer.mozilla.org",
        f"{query} official documentation"
    ]

    all_results = []
    for q in doc_queries:
        results = _tavily_search(q, max_results=3)
        all_results.extend(results)

    # Deduplicate and mark as official_docs
    seen = set()
    doc_results = []
    for r in all_results:
        if r.get("url") not in seen:
            seen.add(r.get("url"))
            r["source_type"] = "official_docs"
            doc_results.append(r)

    return doc_results[:max_results]


def rate_source_quality(url: str, source_name: str = None) -> tuple[str, Optional[str]]:
    """
    Rate the quality of a source based on URL and name.

    Returns: (quality_rating, source_type)
    """
    url_lower = url.lower()
    source_name_lower = (source_name or "").lower()

    # High quality sources
    high_quality_domains = [
        "arxiv.org", "IEEE.org", "ACM.org", "Nature.com", "ScienceDirect.com",
        "docs.python.org", "docs.microsoft.com", "developer.mozilla.org",
        "kubernetes.io", "tensorflow.org", "pytorch.org", "numpy.org",
        "pandas.pydata.org", "scikit-learn.org", "LangChain.dev", "LangGraph.ai"
    ]

    # Academic sources
    academic_domains = ["arxiv.org", "pubmed.ncbi.nlm.nih.gov", "scholar.google.com"]

    # News/Blog sources
    medium_quality = ["medium.com", "towardsdatascience.com", "blog.python.org",
                     "engineering.fb.com", "ai.googleblog.com", "openai.com/blog"]

    for domain in high_quality_domains:
        if domain in url_lower:
            if any(acad in url_lower for acad in academic_domains):
                return "high", "academic"
            return "high", "official_docs"

    for domain in academic_domains:
        if domain in url_lower:
            return "high", "academic"

    for domain in medium_quality:
        if domain in url_lower:
            return "medium", "blog"

    # Check for community resources
    community_domains = ["github.com", "stackoverflow.com", "reddit.com",
                        "discuss.pytorch.org", "discord.com"]
    for domain in community_domains:
        if domain in url_lower:
            return "medium", "community"

    # Check for news
    news_domains = ["reuters.com", "bloomberg.com", "techcrunch.com",
                   "venturebeat.com", "theverge.com"]
    for domain in news_domains:
        if domain in url_lower:
            return "medium", "news"

    return "low", None


def multi_source_research(query: str, enable_arxiv: bool = True,
                          enable_docs: bool = True) -> List[EvidenceItem]:
    """
    Perform research across multiple sources and compile results.
    """
    all_results: List[EvidenceItem] = []

    # 1. Tavily web search (always)
    tavily_results = _tavily_search(query, max_results=8)
    for r in tavily_results:
        quality, source_type = rate_source_quality(r.get("url"), r.get("source"))
        all_results.append(EvidenceItem(
            title=r.get("title", ""),
            url=r.get("url", ""),
            snippet=r.get("snippet", ""),
            source=r.get("source", "web"),
            source_quality=quality,
            source_type=source_type,
        ))

    # 2. arXiv search (if enabled)
    if enable_arxiv:
        arxiv_results = _arxiv_search(query, max_results=3)
        for r in arxiv_results:
            all_results.append(EvidenceItem(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("snippet", ""),
                source=r.get("source", "arXiv"),
                source_quality="high",
                source_type="academic",
            ))

    # 3. Documentation search (if enabled)
    if enable_docs:
        docs_results = _docs_search(query, max_results=3)
        for r in docs_results:
            all_results.append(EvidenceItem(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("snippet", ""),
                source=r.get("source", "docs"),
                source_quality="high",
                source_type="official_docs",
            ))

    # Deduplicate by URL
    seen_urls = set()
    unique_results = []
    for item in all_results:
        if item.url and item.url not in seen_urls:
            seen_urls.add(item.url)
            unique_results.append(item)

    return unique_results


EVIDENCE_SCHEMA_EXAMPLE = """{
  "evidence": [
    {
      "title": "Article Title",
      "url": "https://example.com/article",
      "published_at": "2024-01-15",
      "snippet": "Brief description of the content",
      "source": "Source Name"
    }
  ]
}"""

RESEARCH_SYSTEM = create_strict_prompt(
    """You are a research synthesizer.

Given raw web search results, produce EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources.
- Normalize published_at to ISO YYYY-MM-DD if reliably inferable; else null (do NOT guess).
- Keep snippets short.
- Deduplicate by URL.""",
    EVIDENCE_SCHEMA_EXAMPLE,
    "EvidencePack"
)


def research_node(state: State) -> dict:
    """Research node that gathers evidence from web search using multi-source approach."""
    queries = (state.get("queries") or [])[:10]

    # PHASE 2: Use multi-source research
    all_evidence: List[EvidenceItem] = []

    # Determine which sources to use based on mode
    mode = state.get("mode", "closed_book")
    enable_arxiv = mode in ["hybrid", "open_book"]
    enable_docs = mode in ["hybrid", "closed_book"]

    for q in queries:
        results = multi_source_research(
            q,
            enable_arxiv=enable_arxiv,
            enable_docs=enable_docs
        )
        all_evidence.extend(results)

    if not all_evidence:
        return {"evidence": []}

    # Deduplicate by URL
    dedup = {}
    for e in all_evidence:
        if e.url:
            dedup[e.url] = e
    evidence = list(dedup.values())

    # Filter by recency for open_book mode
    if state.get("mode") == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state["recency_days"]))
        evidence = [e for e in evidence if (d := _iso_to_date(e.published_at)) and d >= cutoff]

    # Update status to researching
    status_update = update_metadata(state, status=BlogStatus.RESEARCHING)

    # Calculate source quality stats
    high_quality = sum(1 for e in evidence if e.source_quality == "high")
    academic = sum(1 for e in evidence if e.source_type == "academic")

    # Add audit entry for successful research
    audit_entry = add_audit_entry(
        state,
        operation=OperationType.RESEARCH_COMPLETE,
        node="research",
        details=f"Found {len(evidence)} evidence items ({high_quality} high-quality, {academic} academic)",
        success=True
    )

    return {"evidence": evidence, **audit_entry, **status_update}


# ============================================================
# PHASE 2: Fact-Checking Node
# ============================================================

FACT_CHECK_SYSTEM = create_strict_prompt(
    """You are a fact-checking assistant. Analyze the given text and identify factual claims that need verification.

For each claim:
1. Determine if it's verifiable (some claims are opinions or subjective)
2. If verifiable, check if it's supported by the provided evidence URLs
3. Rate the verification status: "verified", "unverified", or "contradicted"

Return a list of claims with their verification status and any supporting evidence URLs.""",
    """{
  "claims": [
    {
      "claim": "The specific factual claim from the text",
      "verification_status": "verified|unverified|contradicted",
      "evidence_url": "https://example.com/supporting-source",
      "verification_notes": "Brief explanation of the verification"
    }
  ]
}""",
    "FactCheckResult"
)


class FactCheckResult(BaseModel):
    """Result of fact-checking analysis."""
    claims: List[ClaimVerification]


def fact_check_node(state: State) -> dict:
    """
    Fact-checking node that verifies claims in generated content.
    Runs after sections are written but before final merge.
    """
    sections = state.get("sections", [])

    if not sections:
        return {"claim_verifications": []}

    # Combine all sections into full text
    full_text = "\n\n".join([md for _, md in sorted(sections, key=lambda x: x[0])])

    # Get evidence URLs for verification
    evidence = state.get("evidence", [])
    evidence_text = "\n".join([
        f"- {e.title}: {e.url}" for e in evidence if e.url
    ])

    if not evidence_text:
        # No evidence to check against
        return {"claim_verifications": []}

    max_retries = 2
    for attempt in range(max_retries):
        try:
            checker = llm.with_structured_output(FactCheckResult)
            result = checker.invoke(
                [
                    SystemMessage(content=FACT_CHECK_SYSTEM),
                    HumanMessage(
                        content=(
                            f"Text to check:\n{full_text[:3000]}\n\n"
                            f"Evidence URLs:\n{evidence_text}"
                        )
                    ),
                ]
            )

            # Validate output
            if result is None:
                continue

            # Add audit entry
            audit_entry = add_audit_entry(
                state,
                operation=OperationType.RESEARCH_COMPLETE,
                node="fact_check",
                details=f"Fact-checked {len(result.claims)} claims",
                success=True
            )

            return {"claim_verifications": result.claims, **audit_entry}

        except (ValidationError, ValueError, AttributeError) as e:
            logger.warning(f"Fact-check attempt {attempt + 1} failed: {e}")
            continue

    return {"claim_verifications": []}


# ============================================================
# PHASE 3: Technical Content Enhancements
# ============================================================

class DiagramSpec(BaseModel):
    """Specification for a diagram."""
    title: str
    type: Literal["flowchart", "sequence", "class", "architecture", "timeline"]
    mermaid_code: str
    description: Optional[str] = None


class ComparisonMatrix(BaseModel):
    """Comparison matrix for tools/frameworks."""
    title: str
    items: List[str]  # Items being compared
    criteria: List[str]  # Comparison criteria
    matrix: List[List[str]]  # Values for each item-criteria combination


DIAGRAM_SYSTEM = create_strict_prompt(
    """You are a technical diagram generator. Create Mermaid diagrams to visualize concepts.

Generate a Mermaid diagram that helps explain the topic. Possible types:
- flowchart: Process flows, decision trees
- sequence: API calls, user interactions
- class: Object-oriented structures
- architecture: System designs
- timeline: Event sequences

Return a valid Mermaid diagram code that can be rendered directly.""",
    """{
  "title": "Diagram Title",
  "type": "flowchart|sequence|class|architecture|timeline",
  "mermaid_code": "graph TD\\n    A[Start] --> B[End]",
  "description": "Brief description of what the diagram shows"
}""",
    "DiagramSpec"
)


COMPARISON_SYSTEM = create_strict_prompt(
    """You are a technical comparison assistant. Create comparison matrices for tools, frameworks, or approaches.

For the given topic, identify key items to compare and relevant criteria.
Create a matrix showing how each item scores on each criterion.

Use simple ratings: "✓✓✓" (excellent), "✓✓" (good), "✓" (basic), "✗" (not supported), "N/A" (not applicable).""",
    """{
  "title": "Comparison Title",
  "items": ["Item 1", "Item 2"],
  "criteria": ["Criterion 1", "Criterion 2"],
  "matrix": [["✓✓", "✓"], ["✓", "✓✓✓"]]
}""",
    "ComparisonMatrix"
)


def diagram_generator_node(state: State) -> dict:
    """
    Diagram generator node that creates Mermaid diagrams for the blog.
    Analyzes content and generates relevant diagrams.
    """
    plan = state.get("plan")
    if not plan:
        return {"diagrams": []}

    topics_for_diagram = []

    # Find sections that would benefit from diagrams
    for task in plan.tasks:
        if task.requires_code or "architecture" in task.title.lower() or "flow" in task.title.lower():
            topics_for_diagram.append({
                "section": task.title,
                "topic": task.goal,
                "task_id": task.id
            })

    if not topics_for_diagram:
        return {"diagrams": []}

    diagrams = []
    max_retries = 1  # Keep it quick

    for topic in topics_for_diagram[:2]:  # Max 2 diagrams
        for attempt in range(max_retries):
            try:
                generator = llm.with_structured_output(DiagramSpec)
                result = generator.invoke(
                    [
                        SystemMessage(content=DIAGRAM_SYSTEM),
                        HumanMessage(
                            content=f"Section: {topic['section']}\nGoal: {topic['topic']}\nTopic: {state['topic']}"
                        ),
                    ]
                )

                if result and result.mermaid_code:
                    diagrams.append({
                        "task_id": topic["task_id"],
                        "title": result.title,
                        "type": result.type,
                        "mermaid_code": result.mermaid_code,
                        "description": result.description,
                    })
                break
            except Exception as e:
                logger.warning(f"Diagram generation attempt {attempt + 1} failed: {e}")
                continue

    # Add audit entry
    audit_entry = add_audit_entry(
        state,
        operation=OperationType.PLAN_CREATED,
        node="diagram_generator",
        details=f"Generated {len(diagrams)} diagrams",
        success=True
    )

    return {"diagrams": diagrams, **audit_entry}


def comparison_matrix_node(state: State) -> dict:
    """
    Comparison matrix node that generates comparison tables for tools/frameworks.
    """
    plan = state.get("plan")
    if not plan or plan.blog_kind != "comparison":
        return {"comparison_matrices": []}

    max_retries = 1

    for attempt in range(max_retries):
        try:
            generator = llm.with_structured_output(ComparisonMatrix)
            result = generator.invoke(
                [
                    SystemMessage(content=COMPARISON_SYSTEM),
                    HumanMessage(
                        content=f"Topic: {state['topic']}\nCreate a comparison matrix for the main tools/frameworks discussed."
                    ),
                ]
            )

            if result and result.items:
                # Add audit entry
                audit_entry = add_audit_entry(
                    state,
                    operation=OperationType.PLAN_CREATED,
                    node="comparison_matrix",
                    details=f"Generated comparison: {result.title}",
                    success=True
                )

                return {
                    "comparison_matrices": [result.model_dump()],
                    **audit_entry
                }
        except Exception as e:
            logger.warning(f"Comparison generation attempt {attempt + 1} failed: {e}")
            continue

    return {"comparison_matrices": []}


# -----------------------------
# 6) Orchestrator (Plan) - with strict JSON enforcement
# -----------------------------
PLAN_SCHEMA_EXAMPLE = """{
  "blog_title": "Title of the Blog Post",
  "audience": "Target audience description",
  "tone": "Professional/Technical/Friendly",
  "blog_kind": "explainer",
  "constraints": ["constraint1", "constraint2"],
  "depth_level": "intermediate",
  "target_audience_description": "Description of target audience expertise level",
  "tasks": [
    {
      "id": 1,
      "title": "Section Title",
      "goal": "What reader should understand after reading",
      "bullets": ["bullet 1", "bullet 2", "bullet 3"],
      "target_words": 300,
      "tags": ["tag1", "tag2"],
      "requires_research": false,
      "requires_citations": false,
      "requires_code": false,
      "depth_level": "intermediate",
      "code_complexity": 3,
      "requires_prerequisites": []
    }
  ]
}"""

ORCH_SYSTEM = create_strict_prompt(
    """You are a senior technical writer and developer advocate.
Produce a highly actionable outline for a technical blog post.

PHASE 2: TECHNICAL DEPTH TIERS
Choose the appropriate depth_level for the ENTIRE blog:
- "beginner": Conceptual explanations, analogies, minimal code, explain every term
- "intermediate": Practical examples, some code, moderate depth, assumes basic knowledge
- "expert": Deep technical details, full code examples, edge cases, optimization tips

For EACH task, specify:
- depth_level: matching the overall blog level
- code_complexity: 1 (minimal snippet) to 5 (full production-ready code)
- requires_prerequisites: list of topics reader should know before this section

Requirements:
- 5–9 tasks, each with goal + 3–6 bullets + target_words.
- Tags are flexible; do not force a fixed taxonomy.

Grounding:
- closed_book: evergreen, no evidence dependence.
- hybrid: use evidence for up-to-date examples; mark those tasks requires_research=True and requires_citations=True.
- open_book: weekly/news roundup:
  - Set blog_kind="news_roundup"
  - No tutorial content unless requested
  - If evidence is weak, plan should explicitly reflect that (don't invent events).""",
    PLAN_SCHEMA_EXAMPLE,
    "Plan"
)


def orchestrator_node(state: State) -> dict:
    """Orchestrator node that creates the blog plan."""
    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        try:
            planner = llm.with_structured_output(Plan)
            mode = state.get("mode", "closed_book")
            evidence = state.get("evidence", [])
            depth_level = state.get("depth_level", "intermediate")

            forced_kind = "news_roundup" if mode == "open_book" else None

            plan = planner.invoke(
                [
                    SystemMessage(content=ORCH_SYSTEM),
                    HumanMessage(
                        content=(
                            f"Topic: {state['topic']}\n"
                            f"Mode: {mode}\n"
                            f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n"
                            f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n\n"
                            f"=== PHASE 2: DEPTH LEVEL ===\n"
                            f"Target depth level: {depth_level}\n"
                            f"This affects code complexity, explanation style, and prerequisites.\n"
                            f"==============================\n\n"
                            f"Evidence:\n{[e.model_dump() for e in evidence][:16]}"
                        )
                    ),
                ]
            )

            # Update metadata with sections count - now set to PLAN_REVIEW
            status_update = update_metadata(
                state,
                status=BlogStatus.PLAN_REVIEW,
                sections_count=len(plan.tasks)
            ) if plan else {}

            validated = validate_json_output(plan, "Plan")
            if validated is None:
                raise ValueError("Plan output validation failed")

            if forced_kind:
                plan.blog_kind = "news_roundup"

            # Add audit entry for successful plan creation
            audit_entry = add_audit_entry(
                state,
                operation=OperationType.PLAN_CREATED,
                node="orchestrator",
                details=f"Created plan '{plan.blog_title}' with {len(plan.tasks)} tasks for {plan.audience}",
                success=True
            )

            return {"plan": plan, **audit_entry, **status_update}

        except (ValidationError, ValueError, AttributeError) as e:
            last_error = e
            logger.warning(f"Orchestrator attempt {attempt + 1} failed: {e}")
            continue

    # Fallback error if all retries fail
    logger.error(f"All orchestrator retries failed: {last_error}")

    # Add audit entry for failed plan creation
    audit_entry = add_audit_entry(
        state,
        operation=OperationType.PLAN_CREATED,
        node="orchestrator",
        details=f"Plan creation failed after {max_retries} retries",
        success=False,
        error_message=str(last_error)
    )

    raise RuntimeError(f"Failed to generate plan after {max_retries} attempts: {last_error}")


# ============================================================
# PHASE 1.3: Plan Review Node
# ============================================================
def plan_reviewer_node(state: State) -> dict:
    """
    Plan reviewer node that prepares the plan for user approval.
    Sets status to PLAN_REVIEW and awaits user input.
    """
    plan = state.get("plan")
    if plan is None:
        raise ValueError("plan_reviewer_node called without plan.")

    # Update metadata to show plan is ready for review
    status_update = update_metadata(
        state,
        status=BlogStatus.PLAN_REVIEW,
        sections_count=len(plan.tasks)
    )

    # Add audit entry
    audit_entry = add_audit_entry(
        state,
        operation=OperationType.PLAN_CREATED,
        node="plan_reviewer",
        details=f"Plan ready for review: '{plan.blog_title}' with {len(plan.tasks)} tasks",
        success=True
    )

    return {
        "awaiting_plan_approval": True,
        **status_update,
        **audit_entry,
    }


def revision_request_node(state: State) -> dict:
    """
    Handles plan revision requests from users.
    If user provides modified tasks, updates the plan accordingly.
    """
    revision_notes = state.get("plan_revision_notes")
    plan = state.get("plan")

    if not revision_notes:
        # No specific revisions, just regenerate with same topic
        return {"plan_approved": False}

    # Add audit entry for revision request
    audit_entry = add_audit_entry(
        state,
        operation=OperationType.PLAN_REJECTED,
        node="revision_request",
        details=f"Revision requested: {revision_notes[:100]}...",
        success=True
    )

    return {
        "plan_approved": False,
        "awaiting_plan_approval": False,
        **audit_entry,
    }


def check_plan_approval(state: State) -> str:
    """
    Conditional edge to check if plan is approved.
    If plan is approved (or no approval needed), continue to workers.
    If waiting for approval, the graph exits here - frontend handles resume.
    """
    # If plan_approved is True, continue to workers
    if state.get("plan_approved", False):
        return "worker"

    # If awaiting approval (set by frontend after user clicks approve), continue
    if not state.get("awaiting_plan_approval", False):
        return "worker"

    # Otherwise, we're waiting for approval - graph will exit
    # Frontend will handle the pause and resume
    return "worker"


def after_approval(state: State) -> str:
    """
    Determines next step after plan approval check.
    """
    # If plan is approved or we're continuing from approval, go to workers
    if state.get("plan_approved", False) or not state.get("awaiting_plan_approval", False):
        return "worker"
    return "orchestrator"  # Go back to regenerate if revisions needed


# -----------------------------
# 7) Fanout
# -----------------------------
def fanout(state: State):
    assert state["plan"] is not None
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]


# -----------------------------
# 8) Worker - with enhanced prompt for consistent output
# -----------------------------
WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

PHASE 2: TECHNICAL DEPTH TIERS
Follow the depth_level for this section:

DEPTH LEVEL "beginner":
- Use simple language, avoid jargon OR explain all technical terms
- Include analogies and real-world examples
- Minimal code (1-2 lines), focus on concepts
- Explain "why" behind each concept
- Use bullet points and short paragraphs
- No assumption of prior knowledge

DEPTH LEVEL "intermediate":
- Assume reader knows basics, focus on practical application
- Include working code examples (20-50 lines)
- Explain both "what" and "how"
- Include common use cases and best practices
- Reference official documentation

DEPTH LEVEL "expert":
- Technical deep-dive, assume strong background
- Full production-ready code examples (50+ lines)
- Include edge cases, optimization tips, performance considerations
- Reference source code, papers, or RFCs
- Discuss trade-offs and alternatives
- Include advanced patterns and anti-patterns

Constraints:
- Cover ALL bullets in order.
- Target words ±15%.
- Output only section markdown starting with "## <Section Title>".

IMPORTANT: Your output must be clean Markdown that can be directly concatenated with other sections.
Do NOT include any JSON, code blocks (unless specifically requested), or extra formatting.

Code complexity (1-5):
- 1: Pseudo-code or single function call
- 2: Simple snippet (5-10 lines)
- 3: Working example (20-50 lines)
- 4: Production-like code with error handling (50-100 lines)
- 5: Full implementation with tests and edge cases (100+ lines)

Scope guard:
- If blog_kind=="news_roundup", do NOT drift into tutorials (scraping/RSS/how to fetch).
  Focus on events + implications.

Grounding:
- If mode=="open_book": do not introduce any specific event/company/model/funding/policy claim unless supported by provided Evidence URLs.
  For each supported claim, attach a Markdown link ([Source](URL)).
  If unsupported, write "Not found in provided sources."
- If requires_citations==true (hybrid tasks): cite Evidence URLs for external claims.

Code:
- If requires_code==true, include at least one minimal snippet.
"""


def worker_node(payload: dict) -> dict:
    """Worker node that writes a single section of the blog."""
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(
        f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
        for e in evidence[:20]
    )

    # PHASE 2: Include depth level in the prompt
    prerequisites_text = ", ".join(task.requires_prerequisites) if task.requires_prerequisites else "none"

    max_retries = 2
    last_error = None

    for attempt in range(max_retries):
        try:
            response = llm.invoke(
                [
                    SystemMessage(content=WORKER_SYSTEM),
                    HumanMessage(
                        content=(
                            f"Blog title: {plan.blog_title}\n"
                            f"Audience: {plan.audience}\n"
                            f"Tone: {plan.tone}\n"
                            f"Blog kind: {plan.blog_kind}\n"
                            f"Constraints: {plan.constraints}\n"
                            f"Topic: {payload['topic']}\n"
                            f"Mode: {payload.get('mode')}\n"
                            f"As-of: {payload.get('as_of')} (recency_days={payload.get('recency_days')})\n\n"
                            f"=== PHASE 2: DEPTH TIERS ===\n"
                            f"Blog depth level: {plan.depth_level}\n"
                            f"Section depth level: {task.depth_level}\n"
                            f"Code complexity (1-5): {task.code_complexity}\n"
                            f"Prerequisites: {prerequisites_text}\n"
                            f"========================\n\n"
                            f"Section title: {task.title}\n"
                            f"Goal: {task.goal}\n"
                            f"Target words: {task.target_words}\n"
                            f"Tags: {task.tags}\n"
                            f"requires_research: {task.requires_research}\n"
                            f"requires_citations: {task.requires_citations}\n"
                            f"requires_code: {task.requires_code}\n"
                            f"Bullets:{bullets_text}\n\n"
                            f"Evidence (ONLY cite these URLs):\n{evidence_text}\n"
                        )
                    ),
                ]
            )

            if response is None or not hasattr(response, 'content'):
                raise ValueError("Worker returned None or invalid response")

            section_md = response.content.strip()

            # Validate that output starts with ## for proper section heading
            if not section_md.startswith("##"):
                logger.warning(f"Worker output missing section heading, attempt {attempt + 1}")
                # Try to fix by adding heading if missing
                if attempt == 0:
                    continue

            return {"sections": [(task.id, section_md)]}

        except (ValueError, AttributeError) as e:
            last_error = e
            logger.warning(f"Worker attempt {attempt + 1} failed: {e}")
            continue

    # Fallback: return a minimal section if all retries fail
    logger.error(f"All worker retries failed: {last_error}. Using fallback section.")
    fallback_md = f"## {task.title}\n\n{task.goal}\n\n- " + "\n- ".join(task.bullets)
    return {"sections": [(task.id, fallback_md)]}


# ============================================================
# 9) Reducer (subgraph)
#    merge_content
# ============================================================
def merge_content(state: State) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("merge_content called without plan.")
    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"

    # Calculate word count and reading time
    word_count = len(merged_md.split())
    reading_time = max(1, round(word_count / 200, 1))  # Average 200 words/minute

    # Update metadata
    metadata_update = update_metadata(
        state,
        status=BlogStatus.REVIEWING,
        word_count=word_count,
        reading_time_minutes=reading_time
    )

    # Save version
    version_save = save_version(state, merged_md, plan)

    # Add audit entry
    audit_entry = add_audit_entry(
        state,
        operation=OperationType.MERGE_COMPLETE,
        node="reducer",
        details=f"Merged {len(state['sections'])} sections into {word_count} words ({reading_time} min read)",
        success=True
    )

    return {
        "merged_md": merged_md,
        **metadata_update,
        **version_save,
        **audit_entry,
    }


# build reducer subgraph
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", END)
reducer_subgraph = reducer_graph.compile()

# -----------------------------
# 10) Build main graph
# -----------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("plan_reviewer", plan_reviewer_node)  # PHASE 1.3: Plan review node
g.add_node("revision_request", revision_request_node)  # PHASE 1.3: Revision handler
g.add_node("worker", worker_node)
g.add_node("fact_check", fact_check_node)  # PHASE 2: Fact-checking
g.add_node("diagram_generator", diagram_generator_node)  # PHASE 3: Diagrams
g.add_node("comparison_matrix", comparison_matrix_node)  # PHASE 3: Comparisons
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

# PHASE 1.3: Plan review flow
# After orchestrator, go to plan_reviewer to prepare for user approval
g.add_edge("orchestrator", "plan_reviewer")

# After plan_reviewer, the frontend will handle showing the plan for approval
# When frontend resumes with plan_approved=True, we go to workers
g.add_conditional_edges("plan_reviewer", check_plan_approval, {
    "worker": "worker"
})

# After workers complete, go to fact check → diagrams → comparisons → reducer
g.add_edge("worker", "fact_check")  # PHASE 2: Add fact-checking step
g.add_edge("fact_check", "diagram_generator")  # PHASE 3: Generate diagrams
g.add_edge("diagram_generator", "comparison_matrix")  # PHASE 3: Generate comparisons
g.add_edge("comparison_matrix", "reducer")
g.add_edge("reducer", END)

app = g.compile()


# ============================================================
# PHASE 1: Public API for Blog Generation
# ============================================================

def run_blog_generation(topic: str, as_of: str = None) -> dict:
    """
    Run the full blog generation pipeline with enhanced state management.

    Returns a dictionary with:
    - final: The generated blog post
    - metadata: BlogMetadata with stats
    - audit_trail: List of all operations performed
    - versions: List of saved versions
    - plan: The generated plan
    """
    # Create initial state with metadata
    initial_state = create_initial_state(topic, as_of)

    # Run the graph
    result = app.invoke(initial_state)

    # Update final metadata
    final_update = update_metadata(
        result,
        status=BlogStatus.COMPLETED
    )

    # Add final audit entry
    final_audit = add_audit_entry(
        result,
        operation=OperationType.FINAL_OUTPUT,
        node="system",
        details=f"Blog generation completed: {result.get('metadata', {}).word_count} words",
        success=True
    )

    # Merge final updates
    result.update(final_update)
    result.update(final_audit)

    return result


# ============================================================
# PHASE 1.3: Plan Approval API
# ============================================================
def approve_plan(current_state: dict, approved: bool = True, revision_notes: str = None) -> dict:
    """
    Approve or reject the plan and continue blog generation.

    Args:
        current_state: The current state from the graph (after plan_reviewer)
        approved: Whether the plan is approved
        revision_notes: Notes for revision if not approved

    Returns:
        The final blog generation result
    """
    # Update state with approval decision
    state_update = {
        "plan_approved": approved,
        "plan_revision_notes": revision_notes,
        "awaiting_plan_approval": False,
    }

    # If approved, proceed to workers
    if approved:
        # Update metadata to WRITING status
        status_update = update_metadata(
            current_state,
            status=BlogStatus.WRITING
        )

        # Add audit entry for approval
        audit_entry = add_audit_entry(
            current_state,
            operation=OperationType.PLAN_APPROVED,
            node="plan_approval",
            details="Plan approved by user, proceeding to writing",
            success=True
        )

        state_update.update(status_update)
        state_update.update(audit_entry)

        # Continue from plan_reviewer (which will go to workers due to approval)
        # We need to invoke with the updated state
        result = app.invoke(state_update)
    else:
        # Revision requested - go back to orchestrator with notes
        # For now, we'll restart with the same topic (could enhance to pass revision notes)
        audit_entry = add_audit_entry(
            current_state,
            operation=OperationType.PLAN_REJECTED,
            node="plan_approval",
            details=f"Plan rejected, revision requested: {revision_notes[:100] if revision_notes else 'No specific notes'}...",
            success=True
        )

        state_update.update(audit_entry)

        # Return current state with revision flag - frontend should handle restart
        result = {**current_state, **state_update}

    return result


def get_audit_summary(audit_trail: List[AuditEntry]) -> str:
    """Generate a human-readable summary of the audit trail."""
    if not audit_trail:
        return "No operations recorded."

    summary_lines = ["## Audit Trail", ""]
    for entry in audit_trail:
        status = "✓" if entry.success else "✗"
        summary_lines.append(
            f"{status} **{entry.operation.value}** ({entry.node}): {entry.details}"
        )
        if entry.error_message:
            summary_lines.append(f"  Error: {entry.error_message}")

    return "\n".join(summary_lines)


def get_metadata_summary(metadata: BlogMetadata) -> str:
    """Generate a human-readable summary of the metadata."""
    return f"""## Blog Metadata

- **ID**: {metadata.blog_id}
- **Created**: {metadata.created_at}
- **Updated**: {metadata.updated_at}
- **Status**: {metadata.status.value}
- **Version**: {metadata.version}
- **Word Count**: {metadata.word_count}
- **Reading Time**: {metadata.reading_time_minutes} minutes
- **Sections**: {metadata.sections_count}
- **SEO Score**: {metadata.seo_score or "Not calculated"}
- **Readability**: {metadata.readability_score or "Not calculated"}
"""