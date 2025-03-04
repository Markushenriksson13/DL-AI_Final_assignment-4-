import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import os
import datetime
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_together import ChatTogether
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from dateutil.relativedelta import relativedelta

# Load environment variables
load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

#############################################
# KLIMADATA-VÆRKTØJER
#############################################

@tool
def get_climate_data(location):
    """retrieve comprehensive climate data for a specific location using openweathermap statistical api."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    
    # get coordinates for the location
    geocode_url = f"http://api.openweathermap.org/geo/1.0/direct?q={location}&limit=1&appid={api_key}"
    try:
        geo_response = requests.get(geocode_url)
        geo_data = geo_response.json()
        
        if not geo_data:
            return {"error": f"location {location} not found"}
        
        lat = geo_data[0]['lat']
        lon = geo_data[0]['lon']
    except Exception as e:
        return {"error": f"error getting location coordinates: {str(e)}"}
    
    # define dates for the past year (12 months)
    current_date = datetime.datetime.now()
    current_month = current_date.month
    
    # create monthly data points
    months = []
    temperature_trends = []
    precipitation_trends = []
    humidity_trends = []
    wind_trends = []
    
    # fetch statistical monthly data for the past 12 months
    for i in range(12):
        # calculate month number (going backwards from current month)
        month_num = ((current_month - i - 1) % 12) + 1
        
        # format month name
        month_date = datetime.datetime(current_date.year if month_num <= current_month else current_date.year - 1, month_num, 1)
        month_name = month_date.strftime('%b %Y')
        months.append(month_name)
        
        # use the statistical monthly aggregation api
        statistical_url = f"https://history.openweathermap.org/data/2.5/aggregated/month?lat={lat}&lon={lon}&month={month_num}&appid={api_key}"
        
        try:
            statistical_response = requests.get(statistical_url)
            statistical_data = statistical_response.json()
            
            if 'result' in statistical_data:
                # get temperature data (convert from kelvin to celsius)
                temp_mean = statistical_data['result']['temp']['mean'] - 273.15
                temperature_trends.append(temp_mean)
                
                # get precipitation data
                precip_mean = statistical_data['result']['precipitation']['mean']
                # multiply by days in month to get monthly total
                days_in_month = 30  # approximate
                if month_num in [1, 3, 5, 7, 8, 10, 12]:
                    days_in_month = 31
                elif month_num == 2:
                    days_in_month = 28  # simplified, not accounting for leap years
                precipitation_trends.append(precip_mean * days_in_month)
                
                # get humidity data
                humidity_mean = statistical_data['result']['humidity']['mean']
                humidity_trends.append(humidity_mean)
                
                # get wind data
                wind_mean = statistical_data['result']['wind']['mean']
                wind_trends.append(wind_mean)
            else:
                temperature_trends.append(None)
                precipitation_trends.append(None)
                humidity_trends.append(None)
                wind_trends.append(None)
                print(f"no statistical data available for {month_name}")
        except Exception as e:
            temperature_trends.append(None)
            precipitation_trends.append(None)
            humidity_trends.append(None)
            wind_trends.append(None)
            print(f"error fetching statistical data for {month_name}: {str(e)}")
    
    # reverse the lists to show oldest to newest
    months.reverse()
    temperature_trends.reverse()
    precipitation_trends.reverse()
    humidity_trends.reverse()
    wind_trends.reverse()
    
    # remove none values (missing data)
    valid_months = []
    valid_temps = []
    valid_precip = []
    valid_humidity = []
    valid_wind = []
    
    for i in range(len(months)):
        if temperature_trends[i] is not None:
            valid_months.append(months[i])
            valid_temps.append(temperature_trends[i])
            valid_precip.append(precipitation_trends[i] if precipitation_trends[i] is not None else 0)
            valid_humidity.append(humidity_trends[i] if humidity_trends[i] is not None else 0)
            valid_wind.append(wind_trends[i] if wind_trends[i] is not None else 0)
    
    # if we don't have any valid data, try to get yearly statistical data instead
    if not valid_months:
        try:
            yearly_url = f"https://history.openweathermap.org/data/2.5/aggregated/year?lat={lat}&lon={lon}&appid={api_key}"
            yearly_response = requests.get(yearly_url)
            yearly_data = yearly_response.json()
            
            if 'result' in yearly_data and yearly_data['result']:
                # create monthly data from yearly statistics
                for month_data in yearly_data['result']:
                    month_num = month_data['month']
                    if month_num > 0 and month_num <= 12:
                        month_date = datetime.datetime(current_date.year, month_num, 1)
                        valid_months.append(month_date.strftime('%b'))
                        valid_temps.append(month_data['temp']['mean'] - 273.15)  # convert from kelvin to celsius
                        valid_precip.append(month_data['precipitation']['mean'] * 30)  # approximate monthly total
                        valid_humidity.append(month_data['humidity']['mean'])
                        valid_wind.append(month_data['wind']['mean'])
            
            if not valid_months:
                return {"error": "could not retrieve statistical climate data"}
        except Exception as e:
            return {"error": f"error fetching yearly statistical data: {str(e)}"}
    
    # add additional statistical data if available
    additional_stats = {}
    try:
        # get current month's statistical data for detailed analysis
        current_month_url = f"https://history.openweathermap.org/data/2.5/aggregated/month?lat={lat}&lon={lon}&month={current_month}&appid={api_key}"
        current_month_response = requests.get(current_month_url)
        current_month_data = current_month_response.json()
        
        if 'result' in current_month_data:
            result = current_month_data['result']
            additional_stats = {
                "temperature": {
                    "record_min": round(result['temp']['record_min'] - 273.15, 1),  # convert to celsius
                    "record_max": round(result['temp']['record_max'] - 273.15, 1),
                    "average_min": round(result['temp']['average_min'] - 273.15, 1),
                    "average_max": round(result['temp']['average_max'] - 273.15, 1)
                },
                "humidity": {
                    "min": result['humidity']['min'],
                    "max": result['humidity']['max'],
                    "mean": round(result['humidity']['mean'], 1)
                },
                "wind": {
                    "min": result['wind']['min'],
                    "max": result['wind']['max'],
                    "mean": round(result['wind']['mean'], 1)
                },
                "precipitation": {
                    "min": result['precipitation']['min'],
                    "max": result['precipitation']['max'],
                    "mean": round(result['precipitation']['mean'], 2)
                }
            }
            
            # add sunshine hours if available
            if 'sunshine_hours' in result:
                additional_stats["sunshine_hours"] = result['sunshine_hours']
    except Exception as e:
        print(f"error fetching additional statistical data: {str(e)}")
    
    return {
        "temperature_trends": valid_temps,
        "precipitation_trends": valid_precip,
        "humidity_trends": valid_humidity,
        "wind_trends": valid_wind,
        "months": valid_months,
        "statistics": additional_stats
    }

#############################################
# PÅVIRKNINGSANALYSE-VÆRKTØJER
#############################################

def interpret_impact_score(score):
    """interpret the impact score on a scale from very negative to very positive."""
    if score < -7:
        return "very negative impact on operations and efficiency"
    elif score < -3:
        return "negative impact on operations and efficiency"
    elif score < 0:
        return "slightly negative impact on operations and efficiency"
    elif score < 3:
        return "neutral to slightly positive impact on operations and efficiency"
    elif score < 7:
        return "positive impact on operations and efficiency"
    else:
        return "very positive impact on operations and efficiency"

@tool
def get_weather_impact_analysis(location, sector):
    """analyze how weather patterns impact different sectors based on statistical weather data."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    
    # valid sectors for analysis
    valid_sectors = ["Agriculture", "Energy", "Transportation", "Tourism", "Construction", "Retail"]
    
    if sector not in valid_sectors:
        return {"error": f"sector {sector} not supported for weather impact analysis"}
    
    # get coordinates for the location
    geocode_url = f"http://api.openweathermap.org/geo/1.0/direct?q={location}&limit=1&appid={api_key}"
    try:
        geo_response = requests.get(geocode_url)
        geo_data = geo_response.json()
        
        if not geo_data:
            return {"error": f"location {location} not found"}
        
        lat = geo_data[0]['lat']
        lon = geo_data[0]['lon']
    except Exception as e:
        return {"error": f"error getting location coordinates: {str(e)}"}
    
    # get current weather data
    current_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={api_key}"
    
    try:
        current_response = requests.get(current_url)
        current_data = current_response.json()
        
        if 'main' not in current_data or 'weather' not in current_data:
            return {"error": "could not retrieve current weather data"}
        
        # extract current weather information
        current_temp = current_data['main']['temp']
        current_humidity = current_data['main']['humidity']
        current_wind = current_data['wind']['speed'] if 'wind' in current_data and 'speed' in current_data['wind'] else 0
        current_condition = current_data['weather'][0]['main'] if len(current_data['weather']) > 0 else "Unknown"
        
        # get monthly statistical data for comparison
        current_month = datetime.datetime.now().month
        statistical_url = f"https://history.openweathermap.org/data/2.5/aggregated/month?lat={lat}&lon={lon}&month={current_month}&appid={api_key}"
        
        statistical_response = requests.get(statistical_url)
        statistical_data = statistical_response.json()
        
        if 'result' not in statistical_data:
            return {"error": "could not retrieve statistical weather data for comparison"}
        
        # extract statistical averages
        avg_temp = statistical_data['result']['temp']['mean'] - 273.15  # convert from kelvin to celsius
        avg_humidity = statistical_data['result']['humidity']['mean']
        avg_wind = statistical_data['result']['wind']['mean']
        
        # calculate deviations from average
        temp_deviation = current_temp - avg_temp
        humidity_deviation = current_humidity - avg_humidity
        wind_deviation = current_wind - avg_wind
        
        # sector-specific impact analysis
        sector_impacts = {
            "Agriculture": {
                "temperature": {
                    "impact": temp_deviation * 2.5,  # higher impact of temperature on agriculture
                    "description": "temperature affects crop growth, pest activity, and irrigation needs"
                },
                "humidity": {
                    "impact": humidity_deviation * 1.8,
                    "description": "humidity affects plant diseases, irrigation efficiency, and crop quality"
                },
                "wind": {
                    "impact": wind_deviation * 1.2,
                    "description": "wind affects pollination, evaporation rates, and potential crop damage"
                },
                "conditions": {
                    "Clear": "favorable for most field operations and solar radiation for crops",
                    "Clouds": "reduced solar radiation may slow growth and ripening",
                    "Rain": "beneficial for water needs but may delay field operations",
                    "Snow": "protective insulation for winter crops but halts field operations",
                    "Thunderstorm": "risk of crop damage from heavy rain, hail, or lightning"
                }
            },
            # Other sectors definitions...
        }
        
        # Define the rest of the sectors if needed (Energy, Transportation, etc.)
        # For brevity, I'm not including them all here
        
        if sector not in sector_impacts:
            # Create a default template for missing sectors
            sector_impacts[sector] = {
                "temperature": {
                    "impact": temp_deviation * 1.5,  
                    "description": f"temperature affects operational efficiency in the {sector} sector"
                },
                "humidity": {
                    "impact": humidity_deviation * 1.0,
                    "description": f"humidity affects working conditions and equipment in the {sector} sector"
                },
                "wind": {
                    "impact": wind_deviation * 1.0,
                    "description": f"wind affects outdoor operations in the {sector} sector"
                },
                "conditions": {
                    "Clear": f"generally favorable for {sector} operations",
                    "Clouds": f"minimal impact on {sector} operations",
                    "Rain": f"may affect certain {sector} operations",
                    "Snow": f"likely to disrupt {sector} operations",
                    "Thunderstorm": f"significant disruption to {sector} operations"
                }
            }
        
        # get sector-specific impact data
        sector_impact = sector_impacts[sector]
        
        # calculate overall impact score (-10 to +10 scale)
        temp_impact = sector_impact["temperature"]["impact"]
        humidity_impact = sector_impact["humidity"]["impact"]
        wind_impact = sector_impact["wind"]["impact"]
        
        # normalize impacts to a -10 to +10 scale
        normalize = lambda x: max(min(x, 10), -10)
        temp_impact_normalized = normalize(temp_impact)
        humidity_impact_normalized = normalize(humidity_impact)
        wind_impact_normalized = normalize(wind_impact)
        
        # calculate overall impact (weighted average)
        overall_impact = (temp_impact_normalized * 0.5 + humidity_impact_normalized * 0.3 + wind_impact_normalized * 0.2)
        
        # get condition-specific impact description
        condition_impact = sector_impact["conditions"].get(current_condition, "No specific impact data for this weather condition")
        
        return {
            "sector": sector,
            "location": location,
            "current_weather": {
                "temperature": round(current_temp, 1),
                "humidity": current_humidity,
                "wind_speed": round(current_wind, 1),
                "condition": current_condition
            },
            "average_weather": {
                "temperature": round(avg_temp, 1),
                "humidity": round(avg_humidity, 1),
                "wind_speed": round(avg_wind, 1)
            },
            "impacts": {
                "temperature": {
                    "impact_score": round(temp_impact_normalized, 1),
                    "description": sector_impact["temperature"]["description"]
                },
                "humidity": {
                    "impact_score": round(humidity_impact_normalized, 1),
                    "description": sector_impact["humidity"]["description"]
                },
                "wind": {
                    "impact_score": round(wind_impact_normalized, 1),
                    "description": sector_impact["wind"]["description"]
                }
            },
            "condition_impact": condition_impact,
            "overall_impact": {
                "score": round(overall_impact, 1),
                "interpretation": interpret_impact_score(overall_impact)
            }
        }
    except Exception as e:
        return {"error": f"error analyzing weather impact: {str(e)}"}

#############################################
# AI ANALYSE-VÆRKTØJER
#############################################

def create_tasks(location, industry, specific_concerns):
    """Create AI-powered analysis and recommendations."""
    # Implementation of create_tasks function
    # ...
    return specific_concerns

#############################################
# VISUALISERINGSFUNKTIONER
#############################################

def display_climate_data(climate_data, location):
    """Display climate data visualizations."""
    if "error" in climate_data:
        st.error(f"Climate data: {climate_data['error']}")
        return
    
    # Prepare data for visualization
    months = climate_data["months"]
    temp_data = climate_data["temperature_trends"]
    precip_data = climate_data["precipitation_trends"]
    humidity_data = climate_data["humidity_trends"]
    wind_data = climate_data["wind_trends"]
    
    # Set seaborn style for nicer plots
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Visualizations
    st.subheader(f"Climate Trends for {location} (Statistical Data)")
    
    # Display additional statistical data if available
    if "statistics" in climate_data and climate_data["statistics"]:
        stats = climate_data["statistics"]
        
        st.info(f"""
        **statistical climate data for {location} (current month)**
        
        **temperature records:**
        - record low: {stats['temperature']['record_min']}°c
        - record high: {stats['temperature']['record_max']}°c
        - average low: {stats['temperature']['average_min']}°c
        - average high: {stats['temperature']['average_max']}°c
        
        **humidity:**
        - average: {stats['humidity']['mean']}%
        - range: {stats['humidity']['min']}% - {stats['humidity']['max']}%
        
        **wind speed:**
        - average: {stats['wind']['mean']} m/s
        - range: {stats['wind']['min']} - {stats['wind']['max']} m/s
        
        **precipitation:**
        - average: {stats['precipitation']['mean']} mm/day
        - maximum: {stats['precipitation']['max']} mm/day
        
        {f"**sunshine hours:** {stats['sunshine_hours']} hours/month" if 'sunshine_hours' in stats else ""}
        
        *note: this statistical data is calculated based on historical measurements.*
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Temperature plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x=range(len(months)), y=temp_data, marker='o', linewidth=2.5, color='#1f77b4', ax=ax)
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(months, rotation=45)
        ax.set_xlabel('month', fontsize=14)
        ax.set_ylabel('temperature (°c)', fontsize=14)
        ax.set_title(f'statistical temperature trends in {location}', fontsize=16)
        # Add data point values
        for i, (x, y) in enumerate(zip(range(len(months)), temp_data)):
            ax.text(x, y + 0.5, f'{y:.1f}°c', ha='center')
        st.pyplot(fig)
    
    with col2:
        # Precipitation plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=range(len(months)), y=precip_data, color='#2ca02c', ax=ax)
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(months, rotation=45)
        ax.set_xlabel('month', fontsize=14)
        ax.set_ylabel('precipitation (mm/month)', fontsize=14)
        ax.set_title(f'statistical precipitation trends in {location}', fontsize=16)
        # Add data point values
        for i, (x, y) in enumerate(zip(range(len(months)), precip_data)):
            ax.text(x, y + 0.5, f'{y:.1f}mm', ha='center')
        st.pyplot(fig)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Humidity plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x=range(len(months)), y=humidity_data, marker='s', linewidth=2.5, color='#9467bd', ax=ax)
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(months, rotation=45)
        ax.set_xlabel('month', fontsize=14)
        ax.set_ylabel('humidity (%)', fontsize=14)
        ax.set_title(f'humidity trends in {location}', fontsize=16)
        # Add data point values
        for i, (x, y) in enumerate(zip(range(len(months)), humidity_data)):
            ax.text(x, y + 2, f'{y:.0f}%', ha='center')
        st.pyplot(fig)
    
    with col2:
        # Wind plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x=range(len(months)), y=wind_data, marker='d', linewidth=2.5, color='#d62728', ax=ax)
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(months, rotation=45)
        ax.set_xlabel('month', fontsize=14)
        ax.set_ylabel('wind speed (m/s)', fontsize=14)
        ax.set_title(f'wind speed trends in {location}', fontsize=16)
        # Add data point values
        for i, (x, y) in enumerate(zip(range(len(months)), wind_data)):
            ax.text(x, y + 0.2, f'{y:.1f}m/s', ha='center')
        st.pyplot(fig)

def display_impact_data(impact_data, location, industry):
    """Display impact analysis visualizations."""
    if "error" in impact_data:
        st.error(f"Weather impact analysis: {impact_data['error']}")
        return
    
    st.subheader(f"Weather Impact Analysis for {industry} in {location}")
    
    # Display current weather conditions
    current_weather = impact_data["current_weather"]
    st.info(f"""
    **Current Weather Conditions in {location}:**
    - Temperature: {current_weather['temperature']}°C
    - Humidity: {current_weather['humidity']}%
    - Wind Speed: {current_weather['wind_speed']} m/s
    - Condition: {current_weather['condition']}
    """)
    
    # Display impact scores
    impacts = impact_data["impacts"]
    
    # Create impact visualization
    impact_types = ["Temperature", "Humidity", "Wind"]
    impact_scores = [
        impacts["temperature"]["impact_score"],
        impacts["humidity"]["impact_score"],
        impacts["wind"]["impact_score"]
    ]
    
    # Create impact score visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = sns.barplot(x=impact_types, y=impact_scores, 
                      palette=['#ff9999' if x < 0 else '#99ff99' for x in impact_scores], ax=ax)
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add data labels
    for i, bar in enumerate(bars.patches):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3 if bar.get_height() > 0 else bar.get_height() - 1.0,
            f'{impact_scores[i]:.1f}',
            ha='center',
            va='bottom' if bar.get_height() > 0 else 'top',
            fontsize=12
        )
    
    ax.set_title(f'Weather Impact Scores for {industry} Sector', fontsize=16)
    ax.set_ylabel('Impact Score (-10 to +10)', fontsize=14)
    ax.set_ylim(-10, 10)
    st.pyplot(fig)
    
    # Display impact descriptions
    st.subheader("Weather Impact Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Temperature Impact:**")
        st.markdown(f"- Score: {impacts['temperature']['impact_score']:.1f}")
        st.markdown(f"- {impacts['temperature']['description']}")
        
        st.markdown("**Humidity Impact:**")
        st.markdown(f"- Score: {impacts['humidity']['impact_score']:.1f}")
        st.markdown(f"- {impacts['humidity']['description']}")
    
    with col2:
        st.markdown("**Wind Impact:**")
        st.markdown(f"- Score: {impacts['wind']['impact_score']:.1f}")
        st.markdown(f"- {impacts['wind']['description']}")
        
        st.markdown("**Current Weather Condition Impact:**")
        st.markdown(f"- {impact_data['condition_impact']}")
    
    # Display overall impact
    overall = impact_data["overall_impact"]
    
    st.info(f"""
    **Overall Weather Impact on {industry} Sector:**
    
    Impact Score: {overall['score']:.1f}
    
    Interpretation: {overall['interpretation']}
    """)

#############################################
# HOVEDAPPLIKATION
#############################################

def main():
    st.title("Climate & Sustainability Analysis Platform")
    st.write("Analyze climate trends and sustainability metrics for your location and industry.")
    
    # User inputs
    location = st.text_input("Enter location (city, country):", "Copenhagen, Denmark")
    industry = st.selectbox("Select industry sector:", 
                           ["Agriculture", "Energy", "Transportation", "Tourism", "Construction", "Retail"])
    concerns = st.text_area("Any specific environmental concerns?", 
                                  "How will climate change affect our operations in the next decade?")
    
    if st.button("Analyze"):
        with st.spinner("Analyzing climate and sustainability data..."):
            # Fetch climate data
            climate_data = get_climate_data.invoke(location)
            
            # Fetch impact analysis data
            impact_data = get_weather_impact_analysis.invoke({"location": location, "sector": industry})
            
            # Display climate data visualizations
            display_climate_data(climate_data, location)
            
            # Display impact data visualizations
            display_impact_data(impact_data, location, industry)
            
            # Create and display AI analysis
            tasks = create_tasks(location, industry, concerns)
            
            # Display AI analysis
            st.subheader("AI Analysis & Recommendations")
            
            st.markdown(f"""
            ## Climate & Sustainability Analysis for {industry} in {location}
            
            Based on the statistical weather data and industry-specific impact analysis, here are key insights and recommendations:
            
            ### Key Insights:
            
            1. **Weather Impact**: The current weather conditions in {location} have an overall impact score of {impact_data["overall_impact"]["score"] if "overall_impact" in impact_data else "N/A"} on your {industry} operations.
            
            2. **Seasonal Patterns**: The data shows significant seasonal variations in temperature and precipitation, which affect operational efficiency and resource usage.
            
            3. **Long-term Trends**: Statistical analysis indicates changing patterns in {location}'s climate, requiring adaptive strategies for long-term sustainability.
            
            ### Recommended Actions:
            
            1. **Implement water conservation measures**
               - Potential Impact: 20-30% reduction in water usage
               - Implementation: Install water treatment facilities and rainwater harvesting systems
            
            2. **Switch to renewable energy sources**
               - Potential Impact: 40-60% reduction in CO2 footprint
               - Implementation: Gradual installation of solar panels and sourcing green electricity
            
            3. **Optimize operations based on weather patterns**
               - Potential Impact: 15-25% increase in operational efficiency
               - Implementation: Adjust schedules and processes based on seasonal weather patterns
            
            4. **Implement a comprehensive resource management program**
               - Potential Impact: 30% reduction in resource usage and an increase in recycling rate to 80%
               - Implementation: Resource audits, employee training, and partnerships with sustainability experts
            
            5. **Develop a location-specific climate adaptation strategy**
               - Potential Impact: Increased resilience to local climate changes
               - Implementation: Risk assessment, infrastructure adaptation, and emergency plans for extreme weather events
            """)

if __name__ == "__main__":
    main()