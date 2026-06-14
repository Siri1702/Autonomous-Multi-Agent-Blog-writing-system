# 📝 BlogCraft AI

> An intelligent multi-agent system for generating high-quality data science blog posts using LangGraph and Large Language Models.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-green)](https://langchain-ai.github.io/langgraph/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.4%2B-red)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 🚀 Overview

**BlogCraft AI** is an autonomous multi-agent blogging system that leverages the power of Large Language Models (LLMs) to generate comprehensive, well-researched data science blog posts. Built on top of LangGraph's state machine architecture, it coordinates specialized agents for research, planning, and content writing.

### Key Features

- **🤖 Intelligent Routing** — Automatically determines whether a topic requires web research or can be answered from the LLM's knowledge
- **🔍 Multi-Source Research** — Gathers evidence from Tavily, arXiv, GitHub, and official documentation
- **📋 Plan Approval Workflow** — Human-in-the-loop: review and approve blog outline before writing begins
- **📋 Smart Planning** — Creates detailed blog outlines with 5-9 sections, each with specific goals and word targets
- **✍️ Parallel Writing** — Generates sections concurrently using worker agents for faster execution
- **📊 Technical Depth Control** — Supports beginner, intermediate, and expert-level content generation
- **🔗 Source Quality Rating** — Automatically rates sources as high/medium/low quality with type classification
- **✓ Fact-Checking** — Verifies claims in generated content against provided evidence
- **📈 Diagram Generation** — Auto-generates Mermaid diagrams for architecture and flow visualizations
- **⚖️ Comparison Matrices** — Generates comparison tables for tools/frameworks in "comparison" type blogs
- **💾 Version Tracking** — Maintains audit trails and version history for all generated content

## 🏗️ Architecture

BlogCraft AI implements a sophisticated **LangGraph state machine** with the following pipeline:

```
┌─────────┐     ┌────────────┐     ┌─────────────┐     ┌────────┐     ┌─────────┐
│ Topic   │────▶│  Router    │────▶│  Research  │────▶│   ORCH │────▶│ Plan    │
│ Input   │     │ (Decision) │     │(Multi-source)   │(Plan)  │     │ Review  │
└─────────┘     └────────────┘     └─────────────┘     └────────┘     └─────────┘
                                                                              │
                                                                              ▼
                                                                       ┌─────────────┐
                                                                       │   Workers   │
                                                                       │  (Parallel) │
                                                                       └─────────────┘
                                                                              │
                    ┌─────────────┐     ┌───────────┐     ┌──────────────┐
                    │   Fact      │────▶│  Diagram  │────▶│  Comparison  │
                    │   Check     │     │  Generator│     │    Matrix    │
                    └─────────────┘     └───────────┘     └──────────────┘
                                            │
                                            ▼
                                      ┌─────────┐
                                      │ Reducer │
                                      │ (Merge) │
                                      └─────────┘
                                            │
                                            ▼
                                      ┌─────────────┐
                                      │ Final Blog │
                                      │   Output   │
                                      └─────────────┘
```

### Core Components

| Component | Description |
|-----------|-------------|
| **Router** | Analyzes the topic and decides the execution mode (`closed_book`, `hybrid`, or `open_book`) |
| **Research** | Multi-source research: Tavily (web), arXiv (academic), GitHub, official docs |
| **Orchestrator** | Creates a detailed `Plan` with 5-9 `Task` objects, each having title, goal, bullets, and target word count |
| **Plan Review** | Human-in-the-loop approval: pause after planning, resume after user approval |
| **Workers** | Run in parallel via LangGraph's `Send` fanout; each writes one section in Markdown |
| **Fact-Check** | Verifies factual claims in generated content against evidence |
| **Diagram Generator** | Auto-generates Mermaid diagrams for visualizations |
| **Comparison Matrix** | Generates comparison tables for tool/framework blogs |
| **Reducer** | Merges all sections into a single blog post with title and formatting |

### Modes of Operation

- **Closed Book** — Evergreen content that doesn't require research (e.g., "Introduction to Neural Networks")
- **Hybrid** — Needs both evergreen knowledge and recent examples/tools (e.g., "Best Python Libraries for Data Science in 2024")
- **Open Book** — News/weekly content requiring fresh research (e.g., "Latest AI Developments This Week")

## 🛠️ Tech Stack

| Category | Technology |
|----------|-------------|
| **Agent Framework** | [LangGraph](https://langchain-ai.github.io/langgraph/) |
| **LLM Integration** | [LangChain Ollama](https://python.langchain.com/v0.2/docs/integrations/chat/ollama/) |
| **Frontend** | [Streamlit](https://streamlit.io/) |
| **Data Validation** | [Pydantic](https://docs.pydantic.dev/) |
| **Web Search** | [Tavily](https://tavily.com/) |
| **Environment** | python-dotenv |

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- [Ollama](https://ollama.ai/) installed with the `minimax-m2.5:cloud` model
- (Optional) Tavily API key for web research

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/blogcraft-ai.git
cd blogcraft-ai
```

2. **Create a virtual environment**

```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

Create a `.env` file in the root directory:

```env
TAVILY_API_KEY=your_api_key_here
```

> **Note:** Without a Tavily API key, the research step will be skipped, and the system will operate in closed_book mode.

5. **Start Ollama**

Make sure Ollama is running with the required model:

```bash
ollama serve
ollama pull minimax-m2.5:cloud
```

## 🎮 Usage

### Running the Application

Start the Streamlit frontend:

```bash
streamlit run bwa_frontend.py
```

The app will open at `http://localhost:8501`.

### Using the Interface

1. **Enter a Topic** — Type your blog topic in the sidebar (e.g., "Introduction to LangGraph for AI agents")
2. **Select Date** — Choose an "as-of" date for research recency
3. **Choose Depth Level** — Select beginner, intermediate, or expert
4. **Generate** — Click "Generate Blog" to start the writing process

The system will:
- Analyze your topic and determine if research is needed
- Gather relevant evidence from the web (if required)
- Create a structured plan with multiple sections
- Write each section with AI-powered content
- Merge everything into a polished blog post

### Generated Output

Each blog post includes:
- Title and introduction
- 5-9 well-structured sections
- Proper headings and formatting
- Citations (when research is used)
- Word count and reading time estimate

## 📁 Project Structure

```
Multi_Agent/
├── bwa_backend.py       # LangGraph workflow, all nodes, schemas
├── bwa_frontend.py     # Streamlit UI
├── requirements.txt    # Python dependencies
├── .env               # Environment variables
├── CLAUDE.md          # Development instructions
└── README.md         # This file
```

## 🔧 Development

### Configuration

The LLM model is configured in `bwa_backend.py` at line 208:

```python
llm = ChatOllama(model="minimax-m2.5:cloud")
```

To change the model, update this line and ensure Ollama has the model pulled.

### Extending the System

The system is designed for extensibility. Key extension points:

- **Add new routing modes** — Modify `router_node` in `bwa_backend.py`
- **Custom research sources** — Extend the `research_node` function
- **New output formats** — Modify the `reducer` subgraph
- **Quality enhancements** — Add new nodes to the graph

## 📈 Capabilities

### Current Features

- ✅ Intelligent topic routing with 3 modes
- ✅ Web research via Tavily API
- ✅ Structured blog planning with task decomposition
- ✅ Parallel section writing
- ✅ Technical depth tiers (beginner/intermediate/expert)
- ✅ Code complexity control (1-5 scale)
- ✅ Version tracking and audit trails
- ✅ Markdown output with proper formatting
- ✅ Streamlit UI with real-time progress

### Planned Enhancements

See the [enhancement plan](./.claude/enhancement_plan.md) for upcoming features including:

- Code execution and verification
- Diagram generation (Mermaid)
- SEO optimization
- Fact-checking node
- Multi-perspective analysis
- Export to PDF/HTML

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- [LangChain](https://langchain.ai/) for the LangGraph framework
- [Ollama](https://ollama.ai/) for local LLM inference
- [Tavily](https://tavily.com/) for web search capabilities
- [Streamlit](https://streamlit.io/) for the beautiful UI

---

<p align="center">Built with ❤️ using LangGraph and Streamlit</p>