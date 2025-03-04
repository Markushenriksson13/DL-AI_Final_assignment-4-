import os
from crewai import Agent, Task, Crew, Process
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from langchain.tools import tool
from langchain.chat_models import ChatOpenAI
import requests
import json
from langchain_together import ChatTogether

# Load environment variables
load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Tools for the agents
@tool
def get_climate_data(location):
    """Retrieve climate data for a specific location."""
    # In a real application, this would involve API calls
    # Here we simulate data for demo purposes
    return {
        "temperature_trends": [15, 16, 17, 18, 19],
        "precipitation_trends": [80, 75, 70, 65, 60],
        "air_quality_index": [45, 50, 55, 60, 65],
        "years": [2019, 2020, 2021, 2022, 2023]
    }

@tool
def get_sustainability_metrics(industry_sector):
    """Retrieve sustainability metrics for a specific industry sector."""
    # Simulated data
    return {
        "carbon_footprint": [1000, 950, 900, 850, 800],
        "water_usage": [500, 480, 460, 440, 420],
        "waste_generation": [300, 290, 280, 270, 260],
        "years": [2019, 2020, 2021, 2022, 2023]
    }

@tool
def analyze_trend(data, metric):
    """Analyze a trend in the provided data."""
    if isinstance(data, dict) and metric in data:
        values = data[metric]
        avg = sum(values) / len(values)
        trend = "increasing" if values[-1] > values[0] else "decreasing"
        change = ((values[-1] - values[0]) / values[0]) * 100
        return {
            "average": avg,
            "trend": trend,
            "change_percent": change
        }
    return {"error": "Metric not found in data"}

# Definition of the agents
data_collector = Agent(
    role="Data Collector",
    goal="Collect and organize environmental data from various sources",
    backstory="""You are an expert in environmental data with extensive experience in the 
    collection and organization of climate data, emissions data, and other 
    environment-related metrics. Your ability to extract relevant information from 
    various sources is unparalleled.""",
    verbose=True,
    allow_delegation=True,
    tools=[get_climate_data, get_sustainability_metrics],
    llm=ChatTogether(model="together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", temperature=0.7, base_url='https://api.together.xyz/v1/')
)

analysis_agent = Agent(
    role="Data Analyst",
    goal="Perform in-depth analyses of environmental data to identify meaningful trends",
    backstory="""You are an experienced environmental data analyst with expertise in 
    statistical analyses and trend evaluation. You have experience working with 
    climate data and can perform precise analyses that lead to significant insights.""",
    verbose=True,
    allow_delegation=False,
    tools=[analyze_trend],
    llm=ChatTogether(model="together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", temperature=0.7, base_url='https://api.together.xyz/v1/')
)

insight_generator = Agent(
    role="Environmental Interpreter",
    goal="Interpret data analyses to derive meaningful insights about environmental trends",
    backstory="""You are an environmental scientist specializing in the interpretation 
    of climate data and sustainability analyses. Your background in environmental science allows you to contextualize data and understand their broader impacts.""",
    verbose=True,
    allow_delegation=False,
    llm=ChatTogether(model="together_ai/deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free", temperature=0.7, base_url='https://api.together.xyz/v1/')
)

recommendation_agent = Agent(
    role="Sustainability Advisor",
    goal="Develop personalized, actionable recommendations for environmentally friendly practices",
    backstory="""You are an experienced sustainability advisor who helps individuals and 
    organizations adopt eco-friendly practices. Your recommendations are practical, effective, and tailored to the specific needs and contexts of each client.""",
    verbose=True,
    allow_delegation=False,
    llm=ChatTogether(model="together_ai/deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free", temperature=0.7, base_url='https://api.together.xyz/v1/')
)

# Definition of the tasks
def create_tasks(location, industry, specific_concerns):
    data_collection_task = Task(
        description=f"""Collect comprehensive environmental data for {location} and sustainability metrics 
        for the {industry} industry. Focus on climate trends, emissions, and 
        resource-related metrics from the last 5 years. Organize the data in a 
        structured format suitable for further analysis.
        
        Pay special attention to the following user concerns: {specific_concerns}
        
        Your output should include:
        1. A summary of the collected data
        2. Information about the sources
        3. Notes on potential data gaps or limitations
        """,
        agent=data_collector
    )

    analysis_task = Task(
        description="""Analyze the environmental data provided by the data collector. 
        Identify key trends, outliers, and correlations between different 
        environmental metrics. Perform statistical analyses to assess the significance of the trends.
        
        Your output should include:
        1. A quantitative analysis of the key environmental trends
        2. Comparisons with regional or global benchmarks
        3. Identification of areas with significant changes
        """,
        agent=analysis_agent,
        context=[data_collection_task]
    )

    insight_task = Task(
        description="""Interpret the results of the data analysis to derive meaningful 
        insights about environmental trends and their implications. Consider the 
        data in the broader context of environmental science and sustainability research.
        
        Your output should include:
        1. Main insights from the analyses
        2. The broader implications of these trends for the environment and sustainability
        3. Potential future scenarios based on current trends
        """,
        agent=insight_generator,
        context=[analysis_task]
    )

    recommendation_task = Task(
        description=f"""Develop personalized, actionable recommendations for eco-friendly 
        practices based on the insights from the data analysis. Consider the specific 
        context of {location} and the {industry} industry as well as the user's specific concerns.
        
        User concerns: {specific_concerns}
        
        Your output should include:
        1. 3-5 specific, actionable recommendations
        2. Potential impact of each recommendation
        3. Implementation steps or resources for each recommendation
        """,
        agent=recommendation_agent,
        context=[insight_task]
    )

    return [data_collection_task, analysis_task, insight_task, recommendation_task]

# Streamlit App
def main():
    st.set_page_config(
        page_title="EcoSage - AI Agency for Environmental Analysis",
        page_icon="🌿",
        layout="wide"
    )

    st.title("🌿 EcoSage: AI-powered Environmental Analysis and Sustainability Consulting")
    st.markdown("""
    EcoSage leverages specialized AI agents to analyze environmental data, 
    identify trends, and provide personalized sustainability recommendations.
    """)

    with st.sidebar:
        st.header("Input Parameters")
        location = st.text_input("Location (City or Region)", "Berlin")
        industry = st.selectbox(
            "Industry",
            ["Manufacturing", "Technology", "Agriculture", "Energy", "Transportation", "Retail"]
        )
        concerns = st.text_area(
            "Specific environmental concerns or goals",
            "Reduction of CO2 footprint and more efficient water usage"
        )
        generate_button = st.button("Generate Analysis", type="primary")

    if generate_button:
        with st.spinner("The AI agents are working on your environmental analysis..."):
            # Here the actual CrewAI process would run
            # For demo purposes, we simplify this
            
            # Simulated results for demo purposes
            climate_data = get_climate_data(location)
            sustainability_data = get_sustainability_metrics(industry)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"Climate Trends for {location}")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(climate_data["years"], climate_data["temperature_trends"], marker='o', label='Temperature (°C)')
                ax.set_xlabel('Year')
                ax.set_ylabel('Average Temperature (°C)')
                ax.grid(True)
                ax.legend()
                st.pyplot(fig)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(climate_data["years"], climate_data["air_quality_index"], marker='s', color='purple', label='Air Quality Index')
                ax.set_xlabel('Year')
                ax.set_ylabel('Air Quality Index')
                ax.grid(True)
                ax.legend()
                st.pyplot(fig)
            
            with col2:
                st.subheader(f"Sustainability Metrics for {industry}")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(sustainability_data["years"], sustainability_data["carbon_footprint"], marker='o', color='red', label='CO2 Footprint')
                ax.set_xlabel('Year')
                ax.set_ylabel('CO2 Footprint (t)')
                ax.grid(True)
                ax.legend()
                st.pyplot(fig)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(sustainability_data["years"], sustainability_data["water_usage"], marker='s', color='blue', label='Water Usage')
                ax.set_xlabel('Year')
                ax.set_ylabel('Water Usage (m³)')
                ax.grid(True)
                ax.legend()
                st.pyplot(fig)
            
            # Agent results
            st.header("Results of the AI Analysis")
            
            st.subheader("1. Data Collection")
            st.info(f"""
            **Summary of the collected data for {location} and the {industry} industry:**
            
            The data shows a consistent increase in average temperatures by 4°C over the last 5 years, 
            accompanied by a 20% decrease in precipitation. The air quality index has worsened by 44%, 
            indicating increasing air pollution.
            
            In the {industry} industry, there is a positive trend towards reducing the CO2 footprint by 20%, 
            along with a 16% reduction in water usage and a 13% decrease in waste generation.
            """)
            
            st.subheader("2. Data Analysis")
            st.success(f"""
            **Quantitative Analysis of Environmental Trends:**
            
            The statistical analysis shows that the temperature increase in {location} is significant with a p-value of 0.02 
            and is above the global average of 0.8°C over the same period. The decrease in precipitation correlates strongly 
            with the temperature rise (r = -0.85).
            
            The reduction in the CO2 footprint in the {industry} industry is statistically significant (p < 0.01) and 
            exceeds the industry average of 12%. The improvement in water efficiency is also above the industry average.
            """)
            
            st.subheader("3. Insights")
            st.warning(f"""
            **Key Insights and Implications:**
            
            The climate data for {location} indicates an accelerated local warming, which could lead to increased water scarcity 
            and heat stress. The deterioration in air quality poses a growing health risk, especially for vulnerable populations.
            
            The positive developments in the {industry} industry demonstrate that sustainability measures are effective, 
            although the current reduction rates are not sufficient to meet the Paris climate goals. Without additional measures, 
            the industry might only achieve an overall reduction of 35% by 2030, while 45-50% would be necessary.
            """)
            
            st.subheader("4. Recommendations")
            st.success(f"""
            **Personalized Recommendations Based on Your Concerns:**
            
            1. **Implement a closed-loop water system**
               - Potential Impact: 30-40% reduction in water usage
               - Implementation: Install water treatment facilities and rainwater harvesting systems
            
            2. **Switch to renewable energy sources**
               - Potential Impact: 40-60% reduction in CO2 footprint
               - Implementation: Gradual installation of solar panels and sourcing green electricity
            
            3. **Optimize the supply chain for sustainability**
               - Potential Impact: 25-35% reduction in indirect emissions
               - Implementation: Introduce sustainability criteria for suppliers and local sourcing
            
            4. **Implement a comprehensive waste management program**
               - Potential Impact: 30% reduction in waste generation and an increase in recycling rate to 80%
               - Implementation: Waste audits, employee training, and partnerships with recycling companies
            
            5. **Develop a location-specific climate adaptation strategy**
               - Potential Impact: Increased resilience to local climate changes
               - Implementation: Risk assessment, infrastructure adaptation, and emergency plans for extreme weather events
            """)

if __name__ == "__main__":
    main()
