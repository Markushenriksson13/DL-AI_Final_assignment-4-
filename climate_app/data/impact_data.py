import os
from datetime import datetime
import requests
from utils.constants import (VALID_SECTORS, TEMP_HIGH_THRESHOLD, TEMP_LOW_THRESHOLD, 
                           HUMIDITY_HIGH_THRESHOLD, HUMIDITY_LOW_THRESHOLD, WIND_HIGH_THRESHOLD, 
                           WIND_LOW_THRESHOLD, SEVERE_NEGATIVE_IMPACT, NEGATIVE_IMPACT, 
                           SLIGHT_NEGATIVE_IMPACT, NEUTRAL_IMPACT, POSITIVE_IMPACT)
from utils.helpers import handle_api_error
from data.climate_data import get_location_coordinates

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

def get_weather_impact_analysis(location: str, sector: str) -> dict:
    """Analyze how weather patterns impact different sectors based on statistical weather data."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    
    if sector not in VALID_SECTORS:
        return {"error": f"sector {sector} not supported for weather impact analysis"}
    
    # get coordinates for the location
    coordinates, error = get_location_coordinates(location, api_key)
    if coordinates is None:
        return error
    
    lat, lon = coordinates
    
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
