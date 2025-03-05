import os
from langchain.tools import Tool
from langchain_together import ChatTogether
from crewai import Agent
from data.climate_data import get_climate_data
from data.impact_data import get_weather_impact_analysis
from ai.tools import analyze_trend

# Agent configurations
AGENT_CONFIGS = {
    "climate_analyst": {
        "role": "Climate Data Analyst",
        "goal": "Analyze weather patterns and their impacts on different sectors",
        "backstory": """You are a climate data specialist with expertise in analyzing weather 
        patterns and their effects on various industries. Your analysis combines current 
        weather data with historical trends to provide actionable insights.""",
        "model": "together_ai/deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
    },
    "impact_analyst": {
        "role": "Impact Assessment Specialist",
        "goal": "Evaluate the impact of weather conditions on specific industries",
        "backstory": """You are an expert in assessing how weather conditions affect different 
        business sectors. Your expertise helps organizations understand and prepare for 
        weather-related challenges and opportunities.""",
        "model": "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
    },
    "recommendation_specialist": {
        "role": "Weather Advisory Specialist",
        "goal": "Provide actionable recommendations based on weather impacts",
        "backstory": """You are a specialist in developing practical recommendations for 
        organizations based on weather conditions and their impacts. Your advice helps 
        businesses optimize their operations and mitigate weather-related risks.""",
        "model": "together_ai/deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
    }
}

def create_agents():
    """Opret og returner alle agenter til brug i crew"""
    climate_analyst = Agent(
        role=AGENT_CONFIGS["climate_analyst"]["role"],
        goal=AGENT_CONFIGS["climate_analyst"]["goal"],
        backstory=AGENT_CONFIGS["climate_analyst"]["backstory"],
        verbose=True,
        allow_delegation=False,
        tools=[
            Tool(name="get_climate_data", func=get_climate_data, 
                 description="Retrieve comprehensive climate data for a specific location"),
            Tool(name="analyze_trend", func=analyze_trend,
                 description="Analyze trends in provided weather data metrics")
        ],
        llm=ChatTogether(model=AGENT_CONFIGS["climate_analyst"]["model"], temperature=0.7)
    )

    impact_analyst = Agent(
        role=AGENT_CONFIGS["impact_analyst"]["role"],
        goal=AGENT_CONFIGS["impact_analyst"]["goal"],
        backstory=AGENT_CONFIGS["impact_analyst"]["backstory"],
        verbose=True,
        allow_delegation=False,
        tools=[
            Tool(
                name="get_weather_impact_analysis",
                func=get_weather_impact_analysis,
                description="Analyze how weather patterns impact different sectors"
            ),
            Tool(
                name="analyze_trend",
                func=analyze_trend,
                description="Analyze trends in provided weather data metrics"
            )
        ],
        llm=ChatTogether(model="together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", temperature=0.7)
    )

    recommendation_specialist = Agent(
        role=AGENT_CONFIGS["recommendation_specialist"]["role"],
        goal=AGENT_CONFIGS["recommendation_specialist"]["goal"],
        backstory=AGENT_CONFIGS["recommendation_specialist"]["backstory"],
        verbose=True,
        allow_delegation=False,
        llm=ChatTogether(model="together_ai/deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free", temperature=0.7)
    )
    
    return climate_analyst, impact_analyst, recommendation_specialist
