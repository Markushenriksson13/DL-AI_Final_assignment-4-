import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import os
from datetime import datetime, timedelta
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
# CLIMATE DATA TOOLS
#############################################

@tool
def get_climate_data(location):
    """retrieve comprehensive climate data for a specific location using openweathermap statistical api."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    
    if not api_key:
        return {"error": "Missing API keys. Make sure OPENWEATHER_API_KEY is set in the .env file"}
    
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
    current_date = datetime.now()
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
        month_date = datetime(current_date.year if month_num <= current_month else current_date.year - 1, month_num, 1)
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
                        month_date = datetime(current_date.year, month_num, 1)
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
    
    # Get current weather data
    current_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={api_key}"
    try:
        current_response = requests.get(current_url)
        current_data = current_response.json()
        
        if 'main' not in current_data:
            return {"error": "Could not retrieve current weather data"}
            
        current_weather = {
            "current_temperature": current_data['main'].get('temp', 'N/A'),
            "current_feels_like": current_data['main'].get('feels_like', 'N/A'),
            "current_humidity": current_data['main'].get('humidity', 'N/A'),
            "current_pressure": current_data['main'].get('pressure', 'N/A'),
            "current_wind_speed": current_data.get('wind', {}).get('speed', 'N/A')
        }
    except Exception as e:
        current_weather = {}
        print(f"Error fetching current weather: {str(e)}")

    # Get 5 day forecast with 3-hour intervals
    forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&units=metric&appid={api_key}"
    try:
        forecast_response = requests.get(forecast_url)
        forecast_data = forecast_response.json()
        
        hourly_temps = []
        hourly_humidity = []
        hourly_wind = []
        hourly_dates = []
        
        if 'list' in forecast_data:
            for item in forecast_data['list']:
                date = datetime.fromtimestamp(item['dt'])
                hourly_temps.append(item['main']['temp'])
                hourly_humidity.append(item['main']['humidity'])
                hourly_wind.append(item['wind']['speed'])
                hourly_dates.append(date.strftime('%d/%m - %H:%M'))
    except Exception as e:
        print(f"Error fetching forecast data: {str(e)}")
        hourly_temps, hourly_humidity, hourly_wind, hourly_dates = [], [], [], []

    # Get daily forecast data
    daily_url = f"https://api.openweathermap.org/data/2.5/forecast/daily?lat={lat}&lon={lon}&cnt=16&units=metric&appid={api_key}"
    try:
        daily_response = requests.get(daily_url)
        daily_data = daily_response.json()
        
        daily_temps_max = []
        daily_temps_min = []
        daily_humidity = []
        daily_wind = []
        daily_dates = []
        
        if 'list' in daily_data:
            for item in daily_data['list']:
                date = datetime.fromtimestamp(item['dt'])
                daily_temps_max.append(item['temp']['max'])
                daily_temps_min.append(item['temp']['min'])
                daily_humidity.append(item['humidity'])
                daily_wind.append(item['speed'])
                daily_dates.append(date.strftime('%Y-%m-%d'))
    except Exception as e:
        print(f"Error fetching daily forecast: {str(e)}")
        daily_temps_max, daily_temps_min, daily_humidity, daily_wind, daily_dates = [], [], [], [], []

    # Update return statement to include new data
    return {
        "temperature_trends": valid_temps,
        "precipitation_trends": valid_precip,
        "humidity_trends": valid_humidity,
        "wind_trends": valid_wind,
        "months": valid_months,
        "statistics": additional_stats,
        # Add new data
        **current_weather,
        "hourly_temperatures": hourly_temps,
        "hourly_humidity": hourly_humidity,
        "hourly_wind": hourly_wind,
        "hourly_dates": hourly_dates,
        "daily_temps_max": daily_temps_max,
        "daily_temps_min": daily_temps_min,
        "daily_humidity": daily_humidity,
        "daily_wind": daily_wind,
        "daily_dates": daily_dates
    }

#############################################
# IMPACT ANALYSIS TOOLS
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
        current_month = datetime.now().month
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
                    "impact": temp_deviation * (-2.5 if temp_deviation > 0 else -1.5),
                    "description": f"{'High' if temp_deviation > 0 else 'Low'} temperature affecting crop growth and irrigation needs"
                },
                "humidity": {
                    "impact": humidity_deviation * (-1.8 if humidity_deviation > 0 else -1.2),
                    "description": f"{'High' if humidity_deviation > 0 else 'Low'} humidity affecting plant diseases and irrigation"
                },
                "wind": {
                    "impact": wind_deviation * (-1.5 if wind_deviation > 5 else 0.8),
                    "description": f"{'Strong' if wind_deviation > 5 else 'Light'} wind affecting pollination and evaporation"
                },
                "conditions": {
                    "Clear": "optimal conditions for field operations",
                    "Clouds": "suitable conditions for most agricultural activities",
                    "Rain": "beneficial for crop growth but may limit field operations",
                    "Snow": "risk of frost damage to crops",
                    "Thunderstorm": "risk of crop damage and unsafe for field operations",
                    "Mist": "increased disease risk for sensitive crops",
                    "Fog": "limited visibility for agricultural operations"
                }
            },
            "Energy": {
                "temperature": {
                    "impact": temp_deviation * (-1.0 if abs(temp_deviation) > 10 else 0.5),
                    "description": f"Temperature {'reducing' if abs(temp_deviation) > 10 else 'optimizing'} energy efficiency"
                },
                "humidity": {
                    "impact": humidity_deviation * (-0.5),
                    "description": "Humidity affecting cooling efficiency"
                },
                "wind": {
                    "impact": wind_deviation * (1.5 if wind_deviation > 0 else -0.5),
                    "description": f"{'Increased' if wind_deviation > 0 else 'Reduced'} wind energy production"
                },
                "conditions": {
                    "Clear": "optimal for solar energy production",
                    "Clouds": "reduced solar energy generation",
                    "Rain": "reduced solar efficiency, normal wind operations",
                    "Snow": "potential system stress, reduced efficiency",
                    "Thunderstorm": "risk to infrastructure, emergency protocols needed",
                    "Mist": "reduced solar generation efficiency",
                    "Fog": "significant reduction in solar energy production"
                }
            }
        }
        
        # For other sectors, create a default template with basic weather impacts
        if sector not in sector_impacts:
            sector_impacts[sector] = {
                "temperature": {
                    "impact": temp_deviation * (-1.0),
                    "description": f"Temperature affecting operational efficiency in {sector}"
                },
                "humidity": {
                    "impact": humidity_deviation * (-0.5),
                    "description": f"Humidity affecting working conditions in {sector}"
                },
                "wind": {
                    "impact": wind_deviation * (-1.0),
                    "description": f"Wind conditions affecting {sector} operations"
                },
                "conditions": {
                    "Clear": f"optimal conditions for {sector} operations",
                    "Clouds": f"normal operating conditions for {sector}",
                    "Rain": f"some operational adjustments needed in {sector}",
                    "Snow": f"significant impact on {sector} operations",
                    "Thunderstorm": f"severe disruption to {sector} operations",
                    "Mist": f"minor impacts on {sector} visibility",
                    "Fog": f"reduced visibility affecting {sector} operations"
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
# AI ANALYSIS TOOLS
#############################################

def create_tasks(location, industry, specific_concerns):
    """Create AI-powered analysis and recommendations."""
    # Implementation of create_tasks function
    # ...
    return specific_concerns

#############################################
# VISUALIZATION FUNCTIONS
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

    # Detailed temperature trend (3-hourly)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(climate_data["hourly_dates"], climate_data["hourly_temperatures"], 
            marker='o', label='Temperature (°C)', color='red', alpha=0.6, markersize=4)
    ax.fill_between(climate_data["hourly_dates"], 
                    [t-1 for t in climate_data["hourly_temperatures"]], 
                    [t+1 for t in climate_data["hourly_temperatures"]], 
                    color='red', alpha=0.2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (°C)')
    plt.xticks(range(0, len(climate_data["hourly_dates"]), 3), 
              [climate_data["hourly_dates"][i] for i in range(0, len(climate_data["hourly_dates"]), 3)],
              rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    # Daily temperature range visualization
    if climate_data["daily_temps_max"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        dates_display = []
        for d in climate_data["daily_dates"]:
            try:
                date_obj = datetime.strptime(d, '%Y-%m-%d')
                dates_display.append(date_obj.strftime('%d/%m'))
            except (ValueError, TypeError):
                dates_display.append(d)
        
        ax.fill_between(dates_display, 
                        climate_data["daily_temps_min"],
                        climate_data["daily_temps_max"],
                        alpha=0.3, color='red', label='Temperature Range')
        ax.plot(dates_display, climate_data["daily_temps_max"], 
                'r--', label='Max Temperature')
        ax.plot(dates_display, climate_data["daily_temps_min"], 
                'b--', label='Min Temperature')
        ax.set_xlabel('Date')
        ax.set_ylabel('Temperature (°C)')
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)

    # Humidity and wind correlation
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Humidity (%)', color='blue')
    ax1.plot(climate_data["hourly_dates"], climate_data["hourly_humidity"], 
             color='blue', label='Humidity')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Wind Speed (m/s)', color='green')
    ax2.plot(climate_data["hourly_dates"], climate_data["hourly_wind"], 
             color='green', label='Wind Speed')
    ax2.tick_params(axis='y', labelcolor='green')

    plt.xticks(range(0, len(climate_data["hourly_dates"]), 3), 
              [climate_data["hourly_dates"][i] for i in range(0, len(climate_data["hourly_dates"]), 3)],
              rotation=45, ha='right')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    st.pyplot(fig)

    # Weather metrics overview
    st.write("### Current Weather Conditions")
    metrics_col1, metrics_col2 = st.columns(2)
    with metrics_col1:
        st.metric("Feels Like", 
                  f"{climate_data['current_feels_like']}°C" 
                  if climate_data['current_feels_like'] != 'N/A' else 'N/A')
        st.metric("Air Pressure", 
                  f"{climate_data['current_pressure']} hPa" 
                  if climate_data['current_pressure'] != 'N/A' else 'N/A')
    with metrics_col2:
        # Calculate average values for next 24h
        if climate_data["hourly_temperatures"] and len(climate_data["hourly_temperatures"]) >= 8:
            avg_temp = sum(climate_data["hourly_temperatures"][:8]) / 8
            avg_humidity = sum(climate_data["hourly_humidity"][:8]) / 8
            st.metric("Avg. Temp (next 24h)", f"{avg_temp:.1f}°C")
            st.metric("Avg. Humidity (next 24h)", f"{avg_humidity:.1f}%")

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
# MAIN APPLICATION
#############################################

def main():
    st.title("Climate & Sustainability Analysis Platform")
    st.write("Analyze climate trends and sustainability metrics for your location and industry.")
    
    # User inputs
    location = st.text_input("Enter location (city, country):", "Copenhagen, Denmark")
    industry = st.selectbox("Select industry sector:", 
                           ["Agriculture", "Energy", "Transportation", "Tourism", "Construction", "Retail"])
    concerns = st.text_area("Any specific environmental concerns?", 
                           "How will the anticipated weather changes impact our operations over the coming year?")
    
    if st.button("Analyze"):
        with st.spinner("Analyzing weather and climate data..."):
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
            
            if "error" in impact_data:
                st.error(f"Error in impact analysis: {impact_data['error']}")
                return
                
            st.markdown(f"""
            ## Weather & Climate Analysis for {industry} in {location}
            
            Based on statistical weather data and sector-specific impact analysis:
            
            ### Key Insights:
            
            1. **Current Weather Conditions**: 
               - Temperature: {impact_data.get("current_weather", {}).get("temperature", "N/A")}°C 
               - Humidity: {impact_data.get("current_weather", {}).get("humidity", "N/A")}%
               - Wind: {impact_data.get("current_weather", {}).get("wind_speed", "N/A")} m/s
            
            2. **Deviation from Normal**:
               - Temperature: {round(impact_data.get("current_weather", {}).get("temperature", 0) - impact_data.get("average_weather", {}).get("temperature", 0), 1)}°C from average
               - Humidity: {round(impact_data.get("current_weather", {}).get("humidity", 0) - impact_data.get("average_weather", {}).get("humidity", 0))}% from average
               - Wind: {round(impact_data.get("current_weather", {}).get("wind_speed", 0) - impact_data.get("average_weather", {}).get("wind_speed", 0), 1)} m/s from average
            
            ### Sector-Specific Impact:
            
            {get_sector_recommendations(industry, impact_data)}
            """)
            
            # Add context-based guidance
            if "error" not in climate_data and "error" not in impact_data:
                current_temp = climate_data["current_temperature"]
                current_wind = climate_data["current_wind_speed"]
                
                context_guidance = get_context_guidance(
                    industry, 
                    current_temp, 
                    current_wind,
                    impact_data["overall_impact"]["score"]
                )
                
                st.info(context_guidance)

def get_sector_recommendations(industry, impact_data):
    """Generate sector-specific recommendations based on actual weather impact data."""
    try:
        # Verify that we have all required data
        if not all(key in impact_data["current_weather"] for key in ["temperature", "humidity", "wind_speed"]) or \
           not all(key in impact_data["average_weather"] for key in ["temperature", "humidity", "wind_speed"]):
            return "Insufficient weather data available for recommendations."
        
        sector_advice = {
            "Agriculture": {
                "high_temp": "Increase irrigation and monitor crop water stress",
                "low_temp": "Protect sensitive crops from frost damage",
                "high_humidity": "Increase monitoring for fungal diseases",
                "low_humidity": "Implement supplementary irrigation",
                "high_wind": "Protect crops from wind damage",
                "low_wind": "Optimal conditions for spraying and pollination"
            },
            "Energy": {
                "high_temp": "Optimize cooling systems for power generation",
                "low_temp": "Protect water-based systems from freezing",
                "high_humidity": "Monitor insulation and corrosion",
                "low_humidity": "Optimal solar energy generation conditions",
                "high_wind": "Maximize wind energy production",
                "low_wind": "Switch to alternative energy sources"
            }
        }
        
        if industry not in sector_advice:
            return "No sector-specific recommendations available."
        
        temp_dev = impact_data["current_weather"]["temperature"] - impact_data["average_weather"]["temperature"]
        humid_dev = impact_data["current_weather"]["humidity"] - impact_data["average_weather"]["humidity"]
        wind_dev = impact_data["current_weather"]["wind_speed"] - impact_data["average_weather"]["wind_speed"]
        
        recommendations = []
        advice = sector_advice[industry]
        
        if temp_dev > 2:
            recommendations.append(advice["high_temp"])
        elif temp_dev < -2:
            recommendations.append(advice["low_temp"])
            
        if humid_dev > 10:
            recommendations.append(advice["high_humidity"])
        elif humid_dev < -10:
            recommendations.append(advice["low_humidity"])
            
        if wind_dev > 2:
            recommendations.append(advice["high_wind"])
        elif wind_dev < -2:
            recommendations.append(advice["low_wind"])
        
        if not recommendations:
            recommendations.append("Weather conditions are near normal - maintain standard operations.")
        
        return "\n".join([f"- {rec}" for rec in recommendations])
        
    except Exception as e:
        return f"Error generating recommendations: {str(e)}"

def get_context_guidance(industry, temp, wind, impact_score):
    """Provide context-specific guidance based on current conditions."""
    if impact_score < -5:
        return f"⚠️ Critical weather conditions for {industry} sector. Consider implementing emergency measures."
    elif impact_score < 0:
        return f"⚠️ Suboptimal conditions. Follow recommendations above to minimize impact."
    elif impact_score > 5:
        return f"✅ Optimal conditions for {industry} activities. Capitalize on favorable weather."
    else:
        return f"ℹ️ Normal operating conditions for {industry} sector. Maintain standard procedures."

if __name__ == "__main__":
    main()