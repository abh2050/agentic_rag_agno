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

# Configure Gemini
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if not gemini_api_key:
    st.error("Missing GEMINI_API_KEY in environment variables")
    st.stop()
genai.configure(api_key=gemini_api_key)

# Initialize agents with Gemini 2.0 Flash
web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=Gemini(id="gemini-2.0-flash"),
    tools=[DuckDuckGoTools()],
    instructions="Always include sources with direct URLs",
    show_tool_calls=True,
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=Gemini(id="gemini-2.0-flash"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
    instructions="Use markdown tables with ▲/▼ trend indicators",
    show_tool_calls=True,
    markdown=True,
)

agent_team = Agent(
    team=[web_agent, finance_agent],
    model=Gemini(id="gemini-2.0-flash"),
    instructions=[
        "Combine real-time web data with financial metrics",
        "Always include: [Source] annotations for web data",
        "Format currency values as $X.XX",
        "Use 💹 and 🔻 emojis for positive/negative trends"
    ],
    show_tool_calls=True,
    markdown=True,
)

# Streamlit UI Configuration
st.set_page_config(page_title="AI Financial Analyst", page_icon="💸", layout="wide")
st.title("💎 Gemini 2.0 Flash Financial Analyst")
st.markdown("""
    <style>
    .stTextArea textarea {font-size: 18px !important; border-radius: 12px;}
    div[data-baseweb="input"] {border-radius: 12px !important;}
    .stButton button {border-radius: 12px; padding: 10px 25px;}
    </style>
    """, unsafe_allow_html=True)

# Agent Monitoring Section
with st.expander("🔍 Live Agent Activity", expanded=True):
    activity_container = st.container()

# Enhanced Callback System
class LiveAgentMonitor:
    def __init__(self, container):
        self.container = container
        self.session = {
            "thoughts": [],
            "actions": [],
            "results": []
        }

    def on_thought(self, agent_name, thought):
        entry = f"🧠 **{agent_name}**\n```\n{thought}\n```"
        self.session["thoughts"].append(entry)
        with self.container:
            st.markdown(entry)

    def on_action(self, agent_name, action, input_data):
        entry = f"⚡ **{agent_name}** | {action}\n`Input:` {input_data}"
        self.session["actions"].append(entry)
        with self.container:
            st.markdown(entry)

    def on_action_result(self, agent_name, result):
        entry = f"✅ **{agent_name} Result**\n```json\n{result}\n```"
        self.session["results"].append(entry)
        with self.container:
            st.markdown(entry)

monitor = LiveAgentMonitor(activity_container)

# User Input Panel
query = st.text_area(
    "📩 Your Financial Query:",
    "Analyze AMD's competitive position in AI chips market including stock performance and recent news",
    height=130
)

# Control Panel
col1, col2, col3 = st.columns(3)
with col1:
    temp = st.slider("🌡️ Model Creativity", 0.0, 1.0, 0.8)
with col2:
    depth = st.selectbox("🔎 Analysis Depth", ["Quick Scan", "Detailed Report", "Deep Analysis"])
with col3:
    st.markdown("###")
    run_analysis = st.button("🚀 Execute Analysis", use_container_width=True)

if run_analysis:
    if not query.strip():
        st.warning("Please enter a financial query")
    else:
        with st.spinner("🛠️ Deploying AI Analysts..."):
            try:
                response = agent_team.run(
                    query,
                    callbacks=[monitor],
                    temperature=temp,
                    max_tokens=4096 if "Deep" in depth else 2048
                )
                
                st.markdown("---")
                st.subheader("📜 Final Analysis Report")
                st.markdown(response.content.replace("$", "\$"))
                
                st.markdown("---")
                with st.expander("📦 Raw Session Data"):
                    st.json(monitor.session)
                
            except Exception as e:
                st.error(f"❌ Analysis Failed: {str(e)}")
                st.stop()

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666;">
    Gemini 2.0 Flash Financial Analyst v1.0 | Real-time market data + AI analysis
    </div>
    """, unsafe_allow_html=True)
