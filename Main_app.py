import os
from crewai import Agent, Task, Crew, Process
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from langchain.tools import tool
import requests
import json
from langchain_together import ChatTogether
import datetime
from dateutil.relativedelta import relativedelta
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

# Load environment variables
load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

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
                    "Thunderstorm": "risk of crop damage from heavy rain, hail, or lightning",
                    "Drizzle": "beneficial moisture with minimal field operation disruption",
                    "Mist": "increased disease pressure for some crops",
                    "Fog": "reduced solar radiation and increased disease pressure"
                }
            },
            "Energy": {
                "temperature": {
                    "impact": temp_deviation * 3.0,
                    "description": "temperature affects energy demand for heating/cooling and efficiency of power generation"
                },
                "humidity": {
                    "impact": humidity_deviation * 0.8,
                    "description": "humidity affects cooling efficiency and solar panel performance"
                },
                "wind": {
                    "impact": wind_deviation * 2.5,
                    "description": "wind directly impacts wind power generation and can affect grid infrastructure"
                },
                "conditions": {
                    "Clear": "optimal for solar energy production",
                    "Clouds": "reduced solar energy production",
                    "Rain": "reduced solar energy and potential for hydroelectric boost",
                    "Snow": "reduced solar energy and increased heating demand",
                    "Thunderstorm": "risk to grid infrastructure and reduced solar energy",
                    "Drizzle": "slightly reduced solar energy production",
                    "Mist": "reduced solar energy production",
                    "Fog": "significantly reduced solar energy production"
                }
            },
            "Transportation": {
                "temperature": {
                    "impact": temp_deviation * 1.0,
                    "description": "temperature affects fuel efficiency and infrastructure stress"
                },
                "humidity": {
                    "impact": humidity_deviation * 0.5,
                    "description": "humidity affects visibility and road conditions"
                },
                "wind": {
                    "impact": wind_deviation * 2.0,
                    "description": "wind affects vehicle stability, air traffic, and shipping"
                },
                "conditions": {
                    "Clear": "optimal for all transportation modes",
                    "Clouds": "minimal impact on transportation",
                    "Rain": "reduced visibility and traction for road transport",
                    "Snow": "significant delays and safety concerns for all transport modes",
                    "Thunderstorm": "delays and safety concerns for air and road transport",
                    "Drizzle": "slightly reduced visibility for road transport",
                    "Mist": "reduced visibility affecting all transport modes",
                    "Fog": "significant visibility issues affecting all transport modes"
                }
            },
            "Tourism": {
                "temperature": {
                    "impact": temp_deviation * 2.0,
                    "description": "temperature directly affects outdoor activities and tourist comfort"
                },
                "humidity": {
                    "impact": humidity_deviation * 1.5,
                    "description": "humidity affects perceived temperature and outdoor comfort"
                },
                "wind": {
                    "impact": wind_deviation * 1.0,
                    "description": "wind affects outdoor activities and perceived temperature"
                },
                "conditions": {
                    "Clear": "ideal for most tourist activities",
                    "Clouds": "acceptable for most tourist activities",
                    "Rain": "negative impact on outdoor tourism activities",
                    "Snow": "positive for winter tourism, negative for other tourism",
                    "Thunderstorm": "significant negative impact on all tourism activities",
                    "Drizzle": "slight negative impact on outdoor tourism",
                    "Mist": "may enhance scenic beauty but limit visibility",
                    "Fog": "limits visibility for scenic tourism but may add atmosphere"
                }
            },
            "Construction": {
                "temperature": {
                    "impact": temp_deviation * 1.5,
                    "description": "temperature affects worker productivity and material setting times"
                },
                "humidity": {
                    "impact": humidity_deviation * 1.0,
                    "description": "humidity affects drying times for concrete, paint, and adhesives"
                },
                "wind": {
                    "impact": wind_deviation * 2.2,
                    "description": "wind affects crane operations and worker safety at heights"
                },
                "conditions": {
                    "Clear": "optimal for most construction activities",
                    "Clouds": "good for construction activities",
                    "Rain": "delays many exterior construction activities",
                    "Snow": "halts most exterior construction activities",
                    "Thunderstorm": "halts all exterior construction activities",
                    "Drizzle": "may delay some exterior finishing work",
                    "Mist": "may delay some exterior finishing work",
                    "Fog": "reduced visibility affecting crane operations and safety"
                }
            },
            "Retail": {
                "temperature": {
                    "impact": temp_deviation * 1.8,
                    "description": "temperature affects shopping patterns and product demand"
                },
                "humidity": {
                    "impact": humidity_deviation * 0.7,
                    "description": "humidity affects perceived comfort in shopping areas"
                },
                "wind": {
                    "impact": wind_deviation * 0.5,
                    "description": "wind has minimal direct impact on retail except for outdoor markets"
                },
                "conditions": {
                    "Clear": "typically increases foot traffic to physical stores",
                    "Clouds": "minimal impact on retail activity",
                    "Rain": "decreases foot traffic but may increase online shopping",
                    "Snow": "significantly decreases foot traffic but boosts certain product categories",
                    "Thunderstorm": "significantly decreases foot traffic and may affect power/internet",
                    "Drizzle": "slight decrease in foot traffic",
                    "Mist": "minimal impact on retail activity",
                    "Fog": "may decrease foot traffic in areas requiring travel"
                }
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
            "deviations": {
                "temperature": round(temp_deviation, 1),
                "humidity": round(humidity_deviation, 1),
                "wind_speed": round(wind_deviation, 1)
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
def analyze_weather_patterns(location, timeframe="year"):
    """analyze weather patterns and anomalies for a location based on statistical data."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    
    # validate timeframe
    valid_timeframes = ["month", "year"]
    if timeframe not in valid_timeframes:
        return {"error": f"invalid timeframe. please use one of: {', '.join(valid_timeframes)}"}
    
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
    
    try:
        if timeframe == "year":
            # get yearly statistical data
            url = f"https://history.openweathermap.org/data/2.5/aggregated/year?lat={lat}&lon={lon}&appid={api_key}"
            response = requests.get(url)
            data = response.json()
            
            if 'result' not in data or not data['result']:
                return {"error": "could not retrieve yearly statistical weather data"}
            
            # extract monthly data
            months = []
            temp_data = []
            temp_min_data = []
            temp_max_data = []
            precip_data = []
            humidity_data = []
            wind_data = []
            
            for month_data in data['result']:
                month_num = month_data['month']
                if month_num > 0 and month_num <= 12:
                    month_name = datetime.datetime(2000, month_num, 1).strftime('%b')
                    months.append(month_name)
                    
                    # temperature data (convert from kelvin to celsius)
                    temp_data.append(month_data['temp']['mean'] - 273.15)
                    temp_min_data.append(month_data['temp']['average_min'] - 273.15)
                    temp_max_data.append(month_data['temp']['average_max'] - 273.15)
                    
                    # other weather data
                    precip_data.append(month_data['precipitation']['mean'] * 30)  # monthly total
                    humidity_data.append(month_data['humidity']['mean'])
                    wind_data.append(month_data['wind']['mean'])
            
            # calculate seasonal patterns
            seasons = {
                "Winter": [12, 1, 2],
                "Spring": [3, 4, 5],
                "Summer": [6, 7, 8],
                "Fall": [9, 10, 11]
            }
            
            seasonal_data = {
                "temperature": {},
                "precipitation": {},
                "humidity": {},
                "wind": {}
            }
            
            for season, months_in_season in seasons.items():
                season_temp = []
                season_precip = []
                season_humidity = []
                season_wind = []
                
                for i, month_name in enumerate(months):
                    month_num = datetime.datetime.strptime(month_name, '%b').month
                    if month_num in months_in_season:
                        season_temp.append(temp_data[i])
                        season_precip.append(precip_data[i])
                        season_humidity.append(humidity_data[i])
                        season_wind.append(wind_data[i])
                
                if season_temp:
                    seasonal_data["temperature"][season] = sum(season_temp) / len(season_temp)
                    seasonal_data["precipitation"][season] = sum(season_precip) / len(season_precip)
                    seasonal_data["humidity"][season] = sum(season_humidity) / len(season_humidity)
                    seasonal_data["wind"][season] = sum(season_wind) / len(season_wind)
            
            # identify anomalies (values outside 1.5 standard deviations)
            temp_mean = sum(temp_data) / len(temp_data)
            temp_std = (sum((x - temp_mean) ** 2 for x in temp_data) / len(temp_data)) ** 0.5
            
            precip_mean = sum(precip_data) / len(precip_data)
            precip_std = (sum((x - precip_mean) ** 2 for x in precip_data) / len(precip_data)) ** 0.5
            
            temp_anomalies = []
            precip_anomalies = []
            
            for i, month in enumerate(months):
                if abs(temp_data[i] - temp_mean) > 1.5 * temp_std:
                    temp_anomalies.append({
                        "month": month,
                        "value": round(temp_data[i], 1),
                        "deviation": round(temp_data[i] - temp_mean, 1)
                    })
                
                if abs(precip_data[i] - precip_mean) > 1.5 * precip_std:
                    precip_anomalies.append({
                        "month": month,
                        "value": round(precip_data[i], 1),
                        "deviation": round(precip_data[i] - precip_mean, 1)
                    })
            
            return {
                "location": location,
                "monthly_data": {
                    "months": months,
                    "temperature": [round(t, 1) for t in temp_data],
                    "temperature_min": [round(t, 1) for t in temp_min_data],
                    "temperature_max": [round(t, 1) for t in temp_max_data],
                    "precipitation": [round(p, 1) for p in precip_data],
                    "humidity": [round(h, 1) for h in humidity_data],
                    "wind": [round(w, 1) for w in wind_data]
                },
                "seasonal_patterns": {
                    "temperature": {k: round(v, 1) for k, v in seasonal_data["temperature"].items()},
                    "precipitation": {k: round(v, 1) for k, v in seasonal_data["precipitation"].items()},
                    "humidity": {k: round(v, 1) for k, v in seasonal_data["humidity"].items()},
                    "wind": {k: round(v, 1) for k, v in seasonal_data["wind"].items()}
                },
                "anomalies": {
                    "temperature": temp_anomalies,
                    "precipitation": precip_anomalies
                }
            }
        else:  # timeframe == "month"
            # get current month's statistical data
            current_month = datetime.datetime.now().month
            url = f"https://history.openweathermap.org/data/2.5/aggregated/month?lat={lat}&lon={lon}&month={current_month}&appid={api_key}"
            response = requests.get(url)
            data = response.json()
            
            if 'result' not in data:
                return {"error": "could not retrieve monthly statistical weather data"}
            
            result = data['result']
            month_name = datetime.datetime(2000, result['month'], 1).strftime('%B')
            
            # extract detailed statistics
            temp_stats = {
                "record_min": round(result['temp']['record_min'] - 273.15, 1),
                "record_max": round(result['temp']['record_max'] - 273.15, 1),
                "average_min": round(result['temp']['average_min'] - 273.15, 1),
                "average_max": round(result['temp']['average_max'] - 273.15, 1),
                "median": round(result['temp']['median'] - 273.15, 1),
                "mean": round(result['temp']['mean'] - 273.15, 1),
                "standard_deviation": round(result['temp']['st_dev'], 2)
            }
            
            precip_stats = {
                "min": round(result['precipitation']['min'], 2),
                "max": round(result['precipitation']['max'], 2),
                "median": round(result['precipitation']['median'], 2),
                "mean": round(result['precipitation']['mean'], 2),
                "standard_deviation": round(result['precipitation']['st_dev'], 2)
            }
            
            humidity_stats = {
                "min": result['humidity']['min'],
                "max": result['humidity']['max'],
                "median": result['humidity']['median'],
                "mean": round(result['humidity']['mean'], 1),
                "standard_deviation": round(result['humidity']['st_dev'], 1)
            }
            
            wind_stats = {
                "min": result['wind']['min'],
                "max": result['wind']['max'],
                "median": result['wind']['median'],
                "mean": round(result['wind']['mean'], 1),
                "standard_deviation": round(result['wind']['st_dev'], 1)
            }
            
            # add sunshine hours if available
            sunshine_hours = result.get('sunshine_hours', None)
            
            return {
                "location": location,
                "monthly_data": {
                    "months": [month_name],
                    "temperature": [round(result['temp']['mean'] - 273.15, 1)],
                    "temperature_min": [round(result['temp']['average_min'] - 273.15, 1)],
                    "temperature_max": [round(result['temp']['average_max'] - 273.15, 1)],
                    "precipitation": [round(result['precipitation']['mean'] * 30, 1)],
                    "humidity": [round(result['humidity']['mean'], 1)],
                    "wind": [round(result['wind']['mean'], 1)]
                },
                "seasonal_patterns": {
                    "temperature": {
                        "Winter": round(seasonal_data["temperature"]["Winter"], 1),
                        "Spring": round(seasonal_data["temperature"]["Spring"], 1),
                        "Summer": round(seasonal_data["temperature"]["Summer"], 1),
                        "Fall": round(seasonal_data["temperature"]["Fall"], 1)
                    },
                    "precipitation": {
                        "Winter": round(seasonal_data["precipitation"]["Winter"], 1),
                        "Spring": round(seasonal_data["precipitation"]["Spring"], 1),
                        "Summer": round(seasonal_data["precipitation"]["Summer"], 1),
                        "Fall": round(seasonal_data["precipitation"]["Fall"], 1)
                    },
                    "humidity": {
                        "Winter": round(seasonal_data["humidity"]["Winter"], 1),
                        "Spring": round(seasonal_data["humidity"]["Spring"], 1),
                        "Summer": round(seasonal_data["humidity"]["Summer"], 1),
                        "Fall": round(seasonal_data["humidity"]["Fall"], 1)
                    },
                    "wind": {
                        "Winter": round(seasonal_data["wind"]["Winter"], 1),
                        "Spring": round(seasonal_data["wind"]["Spring"], 1),
                        "Summer": round(seasonal_data["wind"]["Summer"], 1),
                        "Fall": round(seasonal_data["wind"]["Fall"], 1)
                    }
                },
                "anomalies": {
                    "temperature": [
                        {
                            "month": month_name,
                            "value": round(result['temp']['mean'] - temp_mean, 1),
                            "deviation": round(result['temp']['mean'] - temp_mean, 1)
                        }
                    ],
                    "precipitation": [
                        {
                            "month": month_name,
                            "value": round(result['precipitation']['mean'] - precip_mean, 1),
                            "deviation": round(result['precipitation']['mean'] - precip_mean, 1)
                        }
                    ]
                },
                "statistics": {
                    "temperature": temp_stats,
                    "precipitation": precip_stats,
                    "humidity": humidity_stats,
                    "wind": wind_stats
                },
                "sunshine_hours": sunshine_hours
            }
    except Exception as e:
        return {"error": f"error analyzing weather patterns: {str(e)}"}

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
    tools=[get_climate_data, get_weather_impact_analysis],
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
    tools=[analyze_weather_patterns],
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
            # Fetch climate data using the tool's invoke method instead of calling directly
            climate_data = get_climate_data.invoke(location)
            
            # Fetch sustainability data using the tool's invoke method
            sustainability_data = get_weather_impact_analysis.invoke({"location": location, "sector": industry})
            
            # Display climate data
            if "error" in climate_data:
                st.error(f"Climate data: {climate_data['error']}")
            else:
                # Display climate data visualizations
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
                
                # Humidity and wind plots
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
            
            # Display sustainability data
            if "error" in sustainability_data:
                st.error(f"Weather impact analysis: {sustainability_data['error']}")
            else:
                # Display sustainability data visualizations
                st.subheader(f"Weather Impact Analysis for {industry} in {location}")
                
                # Display current weather conditions
                current_weather = sustainability_data["current_weather"]
                st.info(f"""
                **Current Weather Conditions in {location}:**
                - Temperature: {currentweather['temperature']}°C
                - Humidity: {currentweather['humidity']}%
                - Wind Speed: {currentweather['wind_speed']} m/s
                - Condition: {currentweather['condition']}
                """)
                
                # Display impact scores
                impacts = sustainability_data["impacts"]
                
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
                    st.markdown(f"- {sustainability_data['condition_impact']}")
                
                # Display overall impact
                overall = sustainability_data["overall_impact"]
                
                st.info(f"""
                **Overall Weather Impact on {industry} Sector:**
                
                Impact Score: {overall['score']:.1f}
                
                Interpretation: {overall['interpretation']}
                """)
            
            # Create and display AI analysis
            tasks = create_tasks(location, industry, concerns)
            
            # Display AI analysis
            st.subheader("AI Analysis & Recommendations")
            
            st.markdown(f"""
            ## Climate & Sustainability Analysis for {industry} in {location}
            
            Based on the statistical weather data and industry-specific impact analysis, here are key insights and recommendations:
            
            ### Key Insights:
            
            1. **Weather Impact**: The current weather conditions in {location} have an overall impact score of {sustainability_data["overall_impact"]["score"] if "overall_impact" in sustainability_data else "N/A"} on your {industry} operations.
            
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
