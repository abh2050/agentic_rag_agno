import os
import google.generativeai as genai
from dotenv import load_dotenv
import streamlit as st
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

# Load environment variables
load_dotenv()

# --- API Configuration Section ---
gemini_api_key = os.environ.get("GEMINI_API_KEY")

if not gemini_api_key:
    st.error("GEMINI_API_KEY not found in environment variables")
    st.stop()

# Configure both global and Agno-specific settings
genai.configure(api_key=gemini_api_key)

# Test API connection first
try:
    test_model = genai.GenerativeModel('gemini-2.0-flash')
    test_response = test_model.generate_content("Connection test")
except Exception as e:
    st.error(f"API Connection Failed: {str(e)}")
    st.stop()

# --- Agent Setup ---
web_agent = Agent(
    name="Web Researcher",
    role="Gather real-time web data",
    model=Gemini(
        id="gemini-2.0-flash",
        api_key=gemini_api_key  # Explicit key passing
    ),
    tools=[DuckDuckGoTools()],
    instructions="Always cite sources with [1] notation",
    show_tool_calls=True,
    markdown=True,
)

finance_agent = Agent(
    name="Financial Analyst",
    role="Analyze stock market data",
    model=Gemini(
        id="gemini-2.0-flash",
        api_key=gemini_api_key  # Explicit key passing
    ),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
    instructions="Format numbers with $ symbols and ▲/▼ indicators",
    show_tool_calls=True,
    markdown=True,
)

agent_team = Agent(
    team=[web_agent, finance_agent],
    model=Gemini(
        id="gemini-2.0-flash",
        api_key=gemini_api_key  # Explicit key passing
    ),
    instructions=[
        "Combine web and financial data for comprehensive analysis",
        "Use 💹 for positive trends and 🔻 for negative ones",
        "Include timestamps for all data points"
    ],
    show_tool_calls=True,
    markdown=True,
)

# --- Streamlit UI ---
st.set_page_config(page_title="AI Financial Analyst", page_icon="💹", layout="wide")
st.title("📈 Gemini 2.0 Flash Financial Analyst")

# Activity Monitor
activity_log = st.expander("🔍 Live Agent Activity", expanded=True)

class ActivityTracker:
    def __init__(self, container):
        self.container = container
        
    def on_thought(self, agent, thought):
        with self.container:
            st.markdown(f"### 🧠 {agent} Thinking")
            st.code(thought)
            
    def on_action(self, agent, action, inputs):
        with self.container:
            st.markdown(f"#### ⚡ {agent} Action: {action}")
            st.json(inputs)
            
    def on_result(self, agent, result):
        with self.container:
            st.markdown(f"#### ✅ {agent} Result")
            st.markdown(result)

tracker = ActivityTracker(activity_log)

# User Interface
query = st.text_area(
    "📝 Enter your financial query:",
    "Analyze NVIDIA's current market position in AI chips including stock performance and recent news",
    height=100
)

if st.button("🚀 Generate Analysis", type="primary"):
    if not query.strip():
        st.warning("Please enter a valid query")
    else:
        with st.spinner("🧠 Analyzing financial data..."):
            try:
                response = agent_team.run(
                    query,
                    callbacks=[tracker],
                    temperature=0.7,
                    max_tokens=3000
                )
                
                st.markdown("---")
                st.subheader("📊 Analysis Report")
                st.markdown(response.content.replace("$", "\$"))
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.stop()

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666;">
    Powered by Gemini 2.0 Flash | Real-time market analysis
    </div>
    """, unsafe_allow_html=True)
