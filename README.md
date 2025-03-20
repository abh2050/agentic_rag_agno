# Agno-Based AI Agents
![Financial Agent](https://i.ytimg.com/vi/XN6dSSx6Ehg/maxresdefault.jpg)

## Overview

This project implements AI agents using the Agno framework to fetch web search results and financial data. The agents leverage OpenAI's GPT-4o model and utilize tools for searching the web and retrieving stock market data.

## About Agno

[Agno](https://github.com/agno-agi/agno) is a lightweight Python library for building multimodal AI agents capable of handling text, images, audio, and video. It simplifies AI agent development by eliminating the need for complex abstractions like chains or graphs.

### Key Features:
- **Performance:** Fast agent instantiation (~2 microseconds) and low memory usage (~3.75 KiB per agent).
- **Model Agnostic:** Supports multiple AI models, allowing flexibility in deployments.
- **Multimodal Support:** Handles text, images, audio, and video inputs/outputs.
- **Multi-Agent Collaboration:** Agents can work together to complete complex tasks.
- **Memory & Knowledge Integration:** Enables contextual responses and knowledge-based reasoning.

## Features of This Project

- **Web Agent:** Searches the web using DuckDuckGo and provides sources.
- **Finance Agent:** Retrieves financial data (stock prices, analyst recommendations, and company information) from Yahoo Finance.
- **Agent Team:** Combines both agents for enriched insights, ensuring sourced data and tabular displays.

## Code Explanation

The script initializes AI agents using Agno and defines their roles and tools:

1. **Load Environment Variables**
   ```python
   from dotenv import load_dotenv
   import os
   load_dotenv()
   os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
   ```
   - Loads environment variables from a `.env` file.
   - Retrieves the OpenAI API key required for AI model execution.

2. **Initialize Web Agent**
   ```python
   from agno.agent import Agent
   from agno.models.openai import OpenAIChat
   from agno.tools.duckduckgo import DuckDuckGoTools
   
   web_agent = Agent(
       name="Web Agent",
       role="Search the web for information",
       model=OpenAIChat(id="gpt-4o"),
       tools=[DuckDuckGoTools()],
       instructions="Always include sources",
       show_tool_calls=True,
       markdown=True,
   )
   ```
   - Creates a **Web Agent** that searches the internet using DuckDuckGo.
   - Uses OpenAI's GPT-4o model to process and generate responses.
   - Ensures sources are included in responses.

3. **Initialize Finance Agent**
   ```python
   from agno.tools.yfinance import YFinanceTools
   
   finance_agent = Agent(
       name="Finance Agent",
       role="Get financial data",
       model=OpenAIChat(id="gpt-4o"),
       tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
       instructions="Use tables to display data",
       show_tool_calls=True,
       markdown=True,
   )
   ```
   - Creates a **Finance Agent** that fetches stock data from Yahoo Finance.
   - Retrieves stock prices, analyst recommendations, and company info.
   - Outputs results in a structured tabular format.

4. **Combine Agents into a Team**
   ```python
   agent_team = Agent(
       team=[web_agent, finance_agent],
       model=OpenAIChat(id="gpt-4o"),
       instructions=["Always include sources", "Use tables to display data"],
       show_tool_calls=True,
       markdown=True,
   )
   ```
   - Creates a **team of agents** that combines web and finance data retrieval capabilities.
   - Provides consistent formatting and structured outputs.

5. **Execute Query**
   ```python
   agent_team.print_response("What's the market outlook and financial performance of AI semiconductor companies?", stream=True)
   ```
   - Sends a query asking for market trends and financial data of AI semiconductor firms.
   - The **Web Agent** fetches relevant articles and reports.
   - The **Finance Agent** retrieves stock performance and analyst insights.
   - Results are presented with sources and tables.
  
   ![Output](https://github.com/abh2050/agentic_rag_agno/blob/main/pic.png)

## Installation

### Prerequisites
- Python 3.8+
- `agno` package
- `dotenv` for managing environment variables
- OpenAI API key

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/abh2050/agentic_rag_agno.git
   cd <repository_directory>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure API Key:
   Create a `.env` file in the root directory and add the following:
   ```bash
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

Run the script to analyze AI semiconductor companies:
```bash
python main.py
```

### Example Query
```python
agent_team.print_response("What's the market outlook and financial performance of AI semiconductor companies?", stream=True)
```
This will:
- Search the web for market insights
- Fetch financial data from Yahoo Finance
- Present structured information with sources and tables

## Customization

- Modify agent roles, tools, and instructions in the script.
- Change model configurations (e.g., switch OpenAI models if needed).
- Extend functionality by adding new tools or modifying responses.

## License

This project is open-source. Ensure compliance with API usage policies.
