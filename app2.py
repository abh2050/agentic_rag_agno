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

# Initialize agents with Gemini 1.5 Flash
web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=Gemini(id="gemini-1.5-flash-latest"),
    tools=[DuckDuckGoTools()],
    instructions="Always include sources with URLs",
    show_tool_calls=True,
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=Gemini(id="gemini-1.5-flash-latest"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
    instructions="Use formatted tables with emojis for financial data",
    show_tool_calls=True,
    markdown=True,
)

agent_team = Agent(
    team=[web_agent, finance_agent],
    model=Gemini(id="gemini-1.5-flash-latest"),
    instructions=[
        "Combine web and financial data for comprehensive analysis",
        "Always cite sources using [1] notation",
        "Use 📈 and 📉 emojis in financial sections"
    ],
    show_tool_calls=True,
    markdown=True,
)

# Streamlit UI setup
st.set_page_config(page_title="Financial Analysis AI", page_icon="💹", layout="wide")
st.title("💰 AI Financial Analyst")
st.markdown("""
    <style>
    .stTextArea textarea {font-size: 18px !important;}
    div[data-baseweb="input"] {border-radius: 10px !important;}
    </style>
    """, unsafe_allow_html=True)

# Agent activity container
agent_activity_container = st.container()

# Enhanced callback handler
class StreamlitCallbackHandler:
    def __init__(self, container):
        self.container = container
        self.thought_count = 0
        self.action_count = 0

    def on_thought(self, agent_name, thought):
        self.thought_count += 1
        with self.container.expander(f"🧠 {agent_name} Thought #{self.thought_count}", expanded=True):
            st.markdown(f"```\n{thought}\n```")

    def on_action(self, agent_name, action, input_data):
        self.action_count += 1
        with self.container.expander(f"⚡ {agent_name} Action #{self.action_count}: {action}", expanded=False):
            st.json(input_data, expanded=False)

    def on_action_result(self, agent_name, result):
        with self.container.expander(f"✅ {agent_name} Result", expanded=False):
            if isinstance(result, dict):
                st.json(result)
            else:
                st.markdown(result)

callback_handler = StreamlitCallbackHandler(agent_activity_container)

# Main interface
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_area(
        "📝 Ask about any company, stock, or market trend:",
        "Analyze NVIDIA's recent stock performance and market position in AI semiconductors",
        height=120
    )

with col2:
    st.markdown("### Settings")
    temperature = st.slider("🧠 Creativity Level", 0.0, 1.0, 0.7)
    detailed_analysis = st.checkbox("🔍 Detailed Report", True)

if st.button("🚀 Start Analysis", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a question")
    else:
        with st.spinner("🔍 Gathering financial insights..."):
            try:
                response = agent_team.run(
                    query,
                    callbacks=[callback_handler],
                    temperature=temperature,
                    max_tokens=4000 if detailed_analysis else 2000
                )
                
                st.subheader("📈 Analysis Report")
                st.markdown("---")
                st.markdown(response.content.replace("$", "\$"))  # Escape dollar signs for LaTeX
                
            except Exception as e:
                st.error(f"⚠️ Analysis failed: {str(e)}")
                st.stop()

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: gray;">
    Powered by Gemini 1.5 Flash • Made with Agno Framework
    </div>
    """, unsafe_allow_html=True)
