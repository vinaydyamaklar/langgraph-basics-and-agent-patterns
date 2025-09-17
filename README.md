# LangGraph Tutorial: From Basics to Advanced Agent Patterns

A comprehensive repo exploring **LangGraph** fundamentals and advanced agent architectures. This tutorial guides you through key concepts in building intelligent, stateful workflows with Large Language Models (LLMs).

## 🎯 Tutorial Overview

This repo is structured to help learn progressive concepts in LangGraph, from basic graph construction to sophisticated agent patterns. It covers both **LangGraph basics** and **standard LLM workflows** that are essential for building AI applications.

**Perfect for**: Developers learning LangGraph, AI engineers building agent systems, and anyone interested in LLM workflows.

## 📁 Tutorial Structure
```
├── 1_langgraph_basics/           # Core LangGraph concepts
│   ├── Basic building blocks are explaind go through the files
├── 2_standard_llm_workflows/     # Production-ready patterns
│   ├── 1_prompt_chaining.py     # Sequential prompt workflows
│   ├── 2_parallelization.py     # Parallel execution patterns
│   ├── 3_routing.py             # Conditional routing
│   ├── 4_orchestrator_worker.py # Orchestrator-worker pattern
│   └── 5_generator_evaluator.py # Generator-evaluator pattern
└── requirements.txt              # Dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Basic understanding of Python and LLMs
- API keys for LLM providers (OpenAI, Groq, etc.)

### Setup

1. **Clone this tutorial:**

   ```bash
   git clone <repository-url>
   cd simple-langcahin-project
   ```

2. **Install dependencies inside a venv:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file with your API keys:

   ```env
   OPENAI_API_KEY=your_openai_key
   GROQ_API_KEY=your_groq_key
   LANGCHAIN_API_KEY=your_langchain_key
   LANGCHAIN_TRACING_v2=true
   LANGCHAIN_PROJECT=your_project_name
   TAVILY_API_KEY=your_tavily_key
   ```

## 🔍 What You'll Learn

By learnin thr files in this repo, you'll be able to build:

- ✅ Intelligent chatbots with tool integration
- ✅ Multi-step reasoning agents
- ✅ Parallel processing workflows
- ✅ Conditional routing systems
- ✅ Orchestrator-worker architectures
- ✅ Generator-evaluator loops

## 📚 Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)

## 🤝 Contributing

Found an error or want to improve the tutorial? Contributions are welcome!

---

**Ready to start your LangGraph journey? Begin with Lesson 1! 🚀**