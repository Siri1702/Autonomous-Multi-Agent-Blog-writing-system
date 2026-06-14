"""
Blog Writing Agent - Streamlit Frontend
A multi-agent LangGraph system for generating data science blog posts.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Tuple, Iterator

import pandas as pd
import streamlit as st

# ============================================================
# Page Configuration (must be first Streamlit command)
# ============================================================
st.set_page_config(
    page_title="Blog Writing Agent",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# Import the compiled LangGraph app
# ============================================================
try:
    from bwa_backend import app, create_initial_state, BlogMetadata, get_metadata_summary, get_audit_summary, BlogStatus, add_audit_entry, update_metadata
except ImportError as e:
    st.error(f"Failed to import the backend app: {e}")
    st.info("Please ensure bwa_backend.py is in the same directory and has been compiled.")
    st.stop()


# ============================================================
# Helper Functions
# ============================================================

def safe_slug(title: str) -> str:
    """Convert title to a safe filename slug."""
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def try_stream(graph_app, inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Stream graph progress if available; else invoke.
    Yields ("updates"/"values"/"final", payload).
    """
    try:
        for step in graph_app.stream(inputs, stream_mode="updates"):
            yield ("updates", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except Exception:
        pass

    try:
        for step in graph_app.stream(inputs, stream_mode="values"):
            yield ("values", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except Exception:
        pass

    out = graph_app.invoke(inputs)
    yield ("final", out)


def extract_latest_state(current_state: Dict[str, Any], step_payload: Any) -> Dict[str, Any]:
    """Extract and merge the latest state from graph updates."""
    if isinstance(step_payload, dict):
        if len(step_payload) == 1 and isinstance(next(iter(step_payload.values())), dict):
            inner = next(iter(step_payload.values()))
            current_state.update(inner)
        else:
            current_state.update(step_payload)
    return current_state


def list_past_blogs() -> List[Path]:
    """Returns .md files in current working directory, newest first."""
    cwd = Path(".")
    files = [p for p in cwd.glob("*.md") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def read_md_file(p: Path) -> str:
    """Read markdown file content."""
    return p.read_text(encoding="utf-8", errors="replace")


def extract_title_from_md(md: str, fallback: str) -> str:
    """Use first '# ' heading as title if present."""
    for line in md.splitlines():
        if line.startswith("# "):
            t = line[2:].strip()
            return t or fallback
    return fallback


# ============================================================
# Initialize Session State
# ============================================================
if "last_out" not in st.session_state:
    st.session_state["last_out"] = None

if "logs" not in st.session_state:
    st.session_state["logs"] = []

if "topic_prefill" not in st.session_state:
    st.session_state["topic_prefill"] = None


# ============================================================
# Main UI Layout
# ============================================================

# Title and description
st.title("📝 Blog Writing Agent")
st.markdown("Generate high-quality data science blog posts using AI-powered research and writing.")

st.divider()

# Sidebar - Blog Generation
with st.sidebar:
    st.header("🚀 Generate New Blog")

    topic = st.text_area(
        "Topic",
        height=120,
        placeholder="Enter your blog topic (e.g., 'Introduction to LangGraph for AI agents')",
        help="Describe the data science topic you want to write about"
    )

    as_of = st.date_input(
        "As-of date",
        value=date.today(),
        help="The date to use for research recency"
    )

    # PHASE 2: Depth Level Selector
    depth_level = st.selectbox(
        "Technical Depth Level",
        options=["beginner", "intermediate", "expert"],
        index=1,
        help="Choose the technical depth level for the blog post"
    )

    if depth_level == "beginner":
        st.caption("Beginner: Simple language, analogies, minimal code")
    elif depth_level == "intermediate":
        st.caption("Intermediate: Practical examples, assumes basic knowledge")
    else:
        st.caption("Expert: Deep technical details, production-ready code")

    st.divider()

    run_btn = st.button("✨ Generate Blog", type="primary", use_container_width=True)

    st.divider()

    # Past blogs section
    st.subheader("📚 Past Blogs")

    past_files = list_past_blogs()
    if not past_files:
        st.caption("No saved blogs found (*.md in current folder).")
        selected_md_file = None
    else:
        # Build labels from file name + parsed title
        options: List[str] = []
        file_by_label: Dict[str, Path] = {}

        for p in past_files[:50]:
            try:
                md_text = read_md_file(p)
                title = extract_title_from_md(md_text, p.stem)
            except Exception:
                title = p.stem
            label = f"{title}  ·  {p.name}"
            options.append(label)
            file_by_label[label] = p

        selected_label = st.selectbox(
            "Select a blog to load",
            options=options,
            label_visibility="collapsed",
        )
        selected_md_file = file_by_label.get(selected_label)

        if st.button("📂 Load Selected Blog", use_container_width=True):
            if selected_md_file:
                md_text = read_md_file(selected_md_file)
                st.session_state["last_out"] = {
                    "plan": None,
                    "evidence": [],
                    "final": md_text,
                }
                st.session_state["topic_prefill"] = extract_title_from_md(md_text, selected_md_file.stem)
                st.rerun()


# Main content area - Tabs
tab_plan, tab_evidence, tab_preview, tab_metadata, tab_logs = st.tabs([
    "🧩 Plan",
    "🔎 Evidence",
    "📝 Markdown Preview",
    "📊 Metadata",
    "🧾 Logs"
])

logs: List[str] = []


def log(msg: str):
    """Add a message to the logs."""
    logs.append(msg)


# ============================================================
# Blog Generation Logic
# ============================================================

# PHASE 1.3: Plan Approval Workflow
# Check if we need to show plan approval UI
plan_needs_approval = False
approval_state = st.session_state.get("approval_state")

if approval_state:
    # Check if we have a plan waiting for approval
    current_metadata = approval_state.get("metadata")
    if current_metadata:
        status_val = current_metadata.status if hasattr(current_metadata, 'status') else current_metadata.get("status", "")
        if status_val == "plan_review":
            plan_needs_approval = True

# Initialize variables for plan review
plan_approved = None
revision_notes = ""

if run_btn:
    if not topic.strip():
        st.warning("Please enter a topic for the blog.")
        st.stop()

    # Clear any previous approval state
    st.session_state.pop("approval_state", None)

    # Create initial state with enhanced metadata management
    inputs = create_initial_state(topic.strip(), as_of.isoformat(), depth_level)

    status = st.status("Running graph…", expanded=True)
    progress_area = st.empty()

    current_state: Dict[str, Any] = dict(inputs)
    last_node = None

    # Stream through the graph
    for kind, payload in try_stream(app, inputs):
        if kind in ("updates", "values"):
            node_name = None
            if isinstance(payload, dict) and len(payload) == 1 and isinstance(next(iter(payload.values())), dict):
                node_name = next(iter(payload.keys()))
            if node_name and node_name != last_node:
                status.write(f"➡️ Node: `{node_name}`")
                last_node = node_name

            current_state = extract_latest_state(current_state, payload)

            # PHASE 1.3: Check if we reached plan review stage
            metadata = current_state.get("metadata")
            if metadata:
                current_status = metadata.status if hasattr(metadata, 'status') else "draft"
                if current_status == "plan_review":
                    # Save state for approval and break the loop
                    st.session_state["approval_state"] = current_state
                    status.update(label="⏸️ Plan Ready for Review", state="running", expanded=True)
                    progress_area.info("📋 Plan created! Please review and approve below.")
                    log("[plan_review] Plan ready for user approval")
                    st.rerun()

            summary = {
                "mode": current_state.get("mode"),
                "needs_research": current_state.get("needs_research"),
                "queries": current_state.get("queries", [])[:5] if isinstance(current_state.get("queries"), list) else [],
                "evidence_count": len(current_state.get("evidence", []) or []),
                "tasks": len((current_state.get("plan") or {}).get("tasks", [])) if isinstance(current_state.get("plan"), dict) else None,
                "sections_done": len(current_state.get("sections", []) or []),
            }
            progress_area.json(summary)

            log(f"[{kind}] {json.dumps(payload, default=str)[:1200]}")

        elif kind == "final":
            out = payload
            st.session_state["last_out"] = out
            st.session_state.pop("approval_state", None)  # Clear approval state
            status.update(label="✅ Done", state="complete", expanded=False)
            log("[final] received final state")


# ============================================================
# PHASE 1.3: Plan Approval UI
# ============================================================
if plan_needs_approval and approval_state:
    st.divider()
    st.subheader("📋 Review and Approve Blog Plan")

    # Display the plan for review
    plan_obj = approval_state.get("plan")
    if plan_obj:
        if hasattr(plan_obj, "model_dump"):
            plan_dict = plan_obj.model_dump()
        elif isinstance(plan_obj, dict):
            plan_dict = plan_obj
        else:
            plan_dict = json.loads(json.dumps(plan_obj, default=str))

        st.write("### 📝 " + plan_dict.get("blog_title", "Untitled"))
        cols = st.columns(3)
        cols[0].metric("Audience", plan_dict.get("audience", "N/A"))
        cols[1].metric("Tone", plan_dict.get("tone", "N/A"))
        cols[2].metric("Type", plan_dict.get("blog_kind", "N/A"))

        # Show tasks in a dataframe
        tasks = plan_dict.get("tasks", [])
        if tasks:
            st.write("#### 📑 Sections")
            df = pd.DataFrame(
                [
                    {
                        "#": t.get("id"),
                        "Section": t.get("title"),
                        "Words": t.get("target_words"),
                        "Goal": t.get("goal", "")[:80] + "...",
                    }
                    for t in tasks
                ]
            ).sort_values("#")
            st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()

    # Approval buttons
    col_approve, col_reject = st.columns(2)

    with col_approve:
        if st.button("✅ Approve Plan", type="primary", use_container_width=True):
            # Continue generation with approval
            # (Imports already loaded at top)

            # Update state with approval
            approval_state["plan_approved"] = True
            approval_state["awaiting_plan_approval"] = False
            approval_state["metadata"].status = BlogStatus.WRITING

            # Add audit entry
            audit_entry = add_audit_entry(
                approval_state,
                type("OperationType", (), {"value": "plan_approved"})(),
                "plan_approval",
                "Plan approved by user, proceeding to writing",
                True
            )
            approval_state.update(audit_entry)

            # Clear approval state and run the rest
            st.session_state.pop("approval_state", None)

            # Stream remaining generation
            status = st.status("Continuing generation…", expanded=True)

            for kind, payload in try_stream(app, approval_state):
                if kind in ("updates", "values"):
                    log(f"[{kind}] {json.dumps(payload, default=str)[:1200]}")
                elif kind == "final":
                    st.session_state["last_out"] = payload
                    status.update(label="✅ Done", state="complete", expanded=False)
                    log("[final] Generation completed after approval")

            st.rerun()

    with col_reject:
        revision_notes = st.text_area(
            "📝 Revision Notes",
            placeholder="Describe what you'd like to change (e.g., 'Add more code examples', 'Focus on beginner level', 'Change tone to more friendly')",
            height=100
        )

        if st.button("❌ Request Revisions", use_container_width=True):
            if not revision_notes:
                st.warning("Please provide revision notes.")
            else:
                # Store revision notes and restart
                st.info("Restarting with your revision notes...")
                # Include revision notes in topic for regeneration
                new_topic = f"{approval_state.get('topic')}. User notes: {revision_notes}"

                # Restart the process
                st.session_state.pop("approval_state", None)

                inputs = create_initial_state(new_topic, as_of.isoformat(), depth_level)

                status = st.status("Regenerating with revisions…", expanded=True)

                for kind, payload in try_stream(app, inputs):
                    if kind in ("updates", "values"):
                        # Check for plan review again
                        current_state = extract_latest_state({}, payload)
                        metadata = current_state.get("metadata")
                        if metadata:
                            current_status = metadata.status if hasattr(metadata, 'status') else "draft"
                            if current_status == "plan_review":
                                st.session_state["approval_state"] = current_state
                                status.update(label="⏸️ Revised Plan Ready", state="running", expanded=True)
                                st.rerun()
                        log(f"[{kind}] {json.dumps(payload, default=str)[:1200]}")
                    elif kind == "final":
                        st.session_state["last_out"] = payload
                        status.update(label="✅ Done", state="complete", expanded=False)

                st.rerun()

    st.divider()
    # Don't show results yet - need approval first
    st.stop()

# ============================================================
# Render Results
# ============================================================

out = st.session_state.get("last_out")

if out:
    # --- Plan Tab ---
    with tab_plan:
        st.subheader("📋 Blog Plan")
        plan_obj = out.get("plan")

        if not plan_obj:
            st.info("No plan found in output.")
        else:
            if hasattr(plan_obj, "model_dump"):
                plan_dict = plan_obj.model_dump()
            elif isinstance(plan_obj, dict):
                plan_dict = plan_obj
            else:
                plan_dict = json.loads(json.dumps(plan_obj, default=str))

            st.write("**Blog Title:**", plan_dict.get("blog_title"))

            cols = st.columns(3)
            cols[0].write("**Audience:** " + str(plan_dict.get("audience")))
            cols[1].write("**Tone:** " + str(plan_dict.get("tone")))
            cols[2].write("**Blog Type:** " + str(plan_dict.get("blog_kind", "")))

            tasks = plan_dict.get("tasks", [])
            if tasks:
                df = pd.DataFrame(
                    [
                        {
                            "ID": t.get("id"),
                            "Title": t.get("title"),
                            "Target Words": t.get("target_words"),
                            "Needs Research": "✓" if t.get("requires_research") else "✗",
                            "Needs Citations": "✓" if t.get("requires_citations") else "✗",
                            "Needs Code": "✓" if t.get("requires_code") else "✗",
                            "Tags": ", ".join(t.get("tags") or []),
                        }
                        for t in tasks
                    ]
                ).sort_values("ID")
                st.dataframe(df, use_container_width=True, hide_index=True)

                with st.expander("📄 View Full Task Details"):
                    st.json(tasks)

    # --- Evidence Tab ---
    with tab_evidence:
        st.subheader("🔍 Research Evidence")
        evidence = out.get("evidence") or []
        mode = out.get("mode", "unknown")
        needs_research = out.get("needs_research", False)

        if not evidence:
            if mode == "closed_book":
                st.info("No evidence needed - 'closed_book' mode was selected (evergreen content, no research required).")
            elif mode == "hybrid":
                st.info("No evidence returned for hybrid mode. This may be because no Tavily API key was configured.")
            elif not needs_research:
                st.info("No evidence needed - research was not required for this topic.")
            else:
                st.info("No evidence returned. This may be because no Tavily API key was configured or the search returned no results.")
            st.caption(f"Mode: {mode} | Needs Research: {needs_research}")
        else:
            # PHASE 2: Show source quality stats
            high_quality = sum(1 for e in evidence if hasattr(e, 'source_quality') and e.source_quality == "high")
            medium_quality = sum(1 for e in evidence if hasattr(e, 'source_quality') and e.source_quality == "medium")
            low_quality = sum(1 for e in evidence if hasattr(e, 'source_quality') and e.source_quality == "low")
            academic = sum(1 for e in evidence if hasattr(e, 'source_type') and e.source_type == "academic")

            cols = st.columns(5)
            cols[0].metric("Total", len(evidence))
            cols[1].metric("High Quality", high_quality)
            cols[2].metric("Medium", medium_quality)
            cols[3].metric("Low", low_quality)
            cols[4].metric("Academic", academic)

            rows = []
            for e in evidence:
                if hasattr(e, "model_dump"):
                    e = e.model_dump()
                # PHASE 2: Add quality badge
                quality = e.get("source_quality", "N/A")
                quality_badge = f"{quality} ⭐" if quality == "high" else (f"{quality} 🔷" if quality == "medium" else quality)
                rows.append({
                    "Title": e.get("title"),
                    "Quality": quality_badge,
                    "Type": e.get("source_type", "N/A"),
                    "Published": e.get("published_at") or "Unknown",
                    "Source": e.get("source"),
                    "URL": e.get("url"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # --- Preview Tab ---
    with tab_preview:
        st.subheader("📝 Generated Blog")
        final_md = out.get("final") or out.get("merged_md") or ""

        if not final_md:
            st.warning("No markdown content found.")
        else:
            # Render the markdown content
            st.markdown(final_md, unsafe_allow_html=False)

            st.divider()

            # Get the blog title for filename
            plan_obj = out.get("plan")
            if hasattr(plan_obj, "blog_title"):
                blog_title = plan_obj.blog_title
            elif isinstance(plan_obj, dict):
                blog_title = plan_obj.get("blog_title", "blog")
            else:
                blog_title = extract_title_from_md(final_md, "blog")

            md_filename = f"{safe_slug(blog_title)}.md"

            # Download button
            st.download_button(
                "⬇️ Download Markdown",
                data=final_md.encode("utf-8"),
                file_name=md_filename,
                mime="text/markdown",
                type="primary"
            )

    # --- Metadata Tab ---
    with tab_metadata:
        st.subheader("📊 Blog Metadata")

        metadata = out.get("metadata")
        if metadata:
            if hasattr(metadata, "model_dump"):
                metadata_dict = metadata.model_dump()
            elif isinstance(metadata, dict):
                metadata_dict = metadata
            else:
                metadata_dict = json.loads(json.dumps(metadata, default=str))

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Word Count", metadata_dict.get("word_count", 0))
                st.metric("Reading Time", f"{metadata_dict.get('reading_time_minutes', 0)} min")
            with col2:
                st.metric("Version", metadata_dict.get("version", 1))
                st.metric("Sections", metadata_dict.get("sections_count", 0))

            st.write("**Status:**", metadata_dict.get("status", "unknown"))
            st.write("**Blog ID:**", metadata_dict.get("blog_id", "N/A"))
            st.write("**Created:**", metadata_dict.get("created_at", "N/A"))
            st.write("**Updated:**", metadata_dict.get("updated_at", "N/A"))
        else:
            st.info("No metadata available.")

        st.divider()
        st.subheader("🧾 Audit Trail")

        audit_trail = out.get("audit_trail") or []
        if audit_trail:
            rows = []
            for entry in audit_trail:
                if hasattr(entry, "model_dump"):
                    entry_dict = entry.model_dump()
                elif isinstance(entry, dict):
                    entry_dict = entry
                else:
                    entry_dict = json.loads(json.dumps(entry, default=str))

                status_icon = "✅" if entry_dict.get("success", True) else "❌"
                rows.append({
                    "Status": status_icon,
                    "Operation": entry_dict.get("operation", ""),
                    "Node": entry_dict.get("node", ""),
                    "Details": entry_dict.get("details", "")[:100],
                    "Time": entry_dict.get("timestamp", "")[:19],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No audit trail available.")

    # --- Logs Tab ---
    with tab_logs:
        st.subheader("📋 Execution Logs")

        if logs:
            st.session_state["logs"].extend(logs)

        st.text_area(
            "Event Log",
            value="\n\n".join(st.session_state["logs"][-80:]),
            height=520,
            label_visibility="collapsed"
        )

else:
    # Welcome message when no blog has been generated
    st.markdown("""
    ### 👋 Welcome to the Blog Writing Agent!

    This AI-powered system helps you create high-quality data science blog posts.

    **How to use:**
    1. Enter a topic in the sidebar
    2. Select an "as-of" date for research recency
    3. Click **Generate Blog** to start the writing process

    The agent will:
    - 📊 Analyze your topic and decide if research is needed
    - 🔍 Gather relevant evidence from the web (if needed)
    - 📝 Create a structured plan with multiple sections
    - ✍️ Write each section with AI-powered content

    **Or load a past blog:**
    - Select a previously generated blog from the sidebar to view it
    """)

    st.info("💡 Tip: The more specific your topic, the better the generated blog will be!")