from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
import os
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# Check if the OpenAI API key is set in the environment
if "OPENAI_API_KEY" not in os.environ:
    st.error("OpenAI API key is not set in the environment. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Initialize agents
web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=OpenAIChat(id="gpt-4"),
    tools=[DuckDuckGoTools()],
    instructions="Always include sources",
    show_tool_calls=True,
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=OpenAIChat(id="gpt-4"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
    instructions="Use tables to display data",
    show_tool_calls=True,
    markdown=True,
)

agent_team = Agent(
    team=[web_agent, finance_agent],
    model=OpenAIChat(id="gpt-4"),
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

# Streamlit UI setup
st.set_page_config(page_title="Financial Analysis AI Assistant", page_icon="📊", layout="wide")
st.title("Financial Analysis AI Assistant")
st.markdown("This app uses AI agents to research financial markets and company performance.")

# Create a container to display agents' actions
agent_activity_container = st.container()

# Custom callback handler to display what the agents are doing
class StreamlitCallbackHandler:
    def __init__(self, container):
        self.container = container

    def on_thought(self, agent_name, thought):
        with self.container.expander(f"🤔 {agent_name} is thinking...", expanded=True):
            st.write(thought)

    def on_action(self, agent_name, action, input_data):
        with self.container.expander(f"⚡ {agent_name} is performing action: {action}", expanded=True):
            st.write(f"🔍 Input: {input_data}")

    def on_action_result(self, agent_name, result):
        with self.container.expander(f"✅ {agent_name} completed action", expanded=True):
            st.write(result)

# Initialize the callback handler
callback_handler = StreamlitCallbackHandler(agent_activity_container)

# User query input
query = st.text_area("What financial information would you like to know?",
                     "What's the market outlook and financial performance of AI semiconductor companies?",
                     height=100)

# Analyze button
if st.button("Analyze"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Analyzing financial data..."):
            # Use agent_team.run() to get the response
            response = agent_team.run(query, callbacks=[callback_handler])
            
            # Display the response in Streamlit
            st.subheader("📊 Analysis Result")
            st.write(response.content)

# Footer
st.markdown("---")
st.markdown("Powered by Agno and OpenAI GPT-4")
