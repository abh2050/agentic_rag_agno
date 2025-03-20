import os
import google.generativeai as genai
from dotenv import load_dotenv
import streamlit as st
import yfinance as yf
import requests
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

# Check if the Gemini API key is set in the environment
if "GEMINI_API_KEY" not in os.environ:
    st.error("Gemini API key is not set in the environment. Please set the GEMINI_API_KEY environment variable.")
    st.stop()

# Configure Gemini Flash Model 2.0
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def gemini_query(prompt):
    """Fetch response from Gemini Flash Model 2.0."""
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text if response else "No response received."

def get_financial_data(ticker):
    """Fetch financial data using Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "Company Name": info.get("longName", "N/A"),
            "Stock Price": info.get("currentPrice", "N/A"),
            "Market Cap": info.get("marketCap", "N/A"),
            "52 Week High": info.get("fiftyTwoWeekHigh", "N/A"),
            "52 Week Low": info.get("fiftyTwoWeekLow", "N/A"),
            "PE Ratio": info.get("trailingPE", "N/A"),
            "Dividend Yield": info.get("dividendYield", "N/A"),
        }
    except Exception as e:
        return {"Error": str(e)}

def search_web(query):
    """Search the web for information using DuckDuckGo."""
    url = f"https://www.duckduckgo.com/html/?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        results = soup.select(".result__snippet")
        return "\n".join([result.text for result in results[:5]]) if results else "No results found."
    return "Failed to retrieve search results."

# Streamlit UI setup
st.set_page_config(page_title="Financial Analysis AI Assistant", page_icon="📊", layout="wide")
st.title("Financial Analysis AI Assistant")
st.markdown("This app uses AI agents to research financial markets and company performance using Google's Gemini Model.")

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
            web_results = search_web(query)
            ai_analysis = gemini_query(f"Provide a detailed financial analysis on: {query}")
            
            st.subheader("📊 Analysis Result")
            st.write(ai_analysis)
            
            st.subheader("🌐 Web Insights")
            st.write(web_results)

# Footer
st.markdown("---")
st.markdown("Powered by Google's Gemini Flash Model 2.0")
