import streamlit as st
import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
import contextlib
import io

# Load environment variables
load_dotenv()

# API key input in sidebar
if "OPENAI_API_KEY" not in os.environ:
    api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        st.warning("Please enter your OpenAI API key to continue.")
        st.stop()

# Initialize agents (cached for performance)
@st.cache_resource
def initialize_agents():
    web_agent = Agent(
        name="Web Agent",
        role="Search the web for information",
        model=OpenAIChat(id="gpt-4o"),
        tools=[DuckDuckGoTools()],
        instructions="Always include sources",
        show_tool_calls=False
    )

    finance_agent = Agent(
        name="Finance Agent",
        role="Get financial data",
        model=OpenAIChat(id="gpt-4o"),
        tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
        instructions="Use tables to display data",
        show_tool_calls=False
    )

    agent_team = Agent(
        team=[web_agent, finance_agent],
        model=OpenAIChat(id="gpt-4o"),
        instructions=["Always include sources", "Use tables to display data"],
        show_tool_calls=False
    )
    return agent_team

# Function to get analysis with suppressed console output
def get_analysis(query):
    agent_team = initialize_agents()
    try:
        # Suppress console output by redirecting stdout
        with contextlib.redirect_stdout(io.StringIO()):
            response = agent_team.get_response(query, stream=False)
        return response
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

# Streamlit UI setup
st.set_page_config(page_title="Financial Analysis AI Assistant", page_icon="📊", layout="wide")

st.title("Financial Analysis AI Assistant")
st.markdown("""
This app uses AI agents to research financial markets and company performance.
Enter your query below to get financial insights powered by web search and financial data.
""")

# Session state to store response and analysis status
if 'response' not in st.session_state:
    st.session_state.response = None
if 'is_analyzing' not in st.session_state:
    st.session_state.is_analyzing = False

# User query input
query = st.text_area("What financial information would you like to know?",
                     "What's the market outlook and financial performance of AI semiconductor companies?",
                     height=100)

# Analyze button
if st.button("Analyze"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        st.session_state.is_analyzing = True
        with st.spinner("Analyzing financial data... This may take a minute."):
            response = get_analysis(query)
            if response:
                st.session_state.response = response
        st.session_state.is_analyzing = False

# Display results in the app
if st.session_state.response:
    st.markdown("### Analysis Results")
    st.markdown(st.session_state.response)

    # Option to download results
    st.download_button(
        label="Download Results",
        data=st.session_state.response,
        file_name="financial_analysis.md",
        mime="text/markdown",
    )

# Footer
st.markdown("---")
st.markdown("Powered by Agno and OpenAI GPT-4o")