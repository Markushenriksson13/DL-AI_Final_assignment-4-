import os
import requests
from datetime import datetime
from utils.helpers import handle_api_error

def get_location_coordinates(location, api_key):
    """Get coordinates for a location"""
    try:
        geocode_url = f"http://api.openweathermap.org/geo/1.0/direct?q={location}&limit=1&appid={api_key}"
        geo_response = requests.get(geocode_url)
        geo_data = geo_response.json()
        
        if not geo_data:
            return None, handle_api_error(f"location {location} not found")
        
        return (geo_data[0]['lat'], geo_data[0]['lon']), None
    except Exception as e:
        return None, handle_api_error("error getting location coordinates", e)

def get_climate_data(location: str) -> dict:
    """Retrieve comprehensive climate data for a specific location using OpenWeatherMap statistical API."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    
    if not api_key:
        return {"error": "Missing API keys. Make sure OPENWEATHER_API_KEY is set in the .env file"}
    
    # get coordinates for the location
    coordinates, error = get_location_coordinates(location, api_key)
    if coordinates is None:
        return error
    
    lat, lon = coordinates
    
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

    # Fix for daily forecast: Use standard forecast API data to generate daily values
    daily_temps_max = []
    daily_temps_min = []
    daily_humidity = []
    daily_wind = []
    daily_dates = []
    
    try:
        if 'list' in forecast_data:
            # Group forecast data by day
            daily_data = {}
            for item in forecast_data['list']:
                date = datetime.fromtimestamp(item['dt'])
                day_key = date.strftime('%Y-%m-%d')
                
                if day_key not in daily_data:
                    daily_data[day_key] = {
                        'temps': [],
                        'humidity': [],
                        'wind': []
                    }
                
                daily_data[day_key]['temps'].append(item['main']['temp'])
                daily_data[day_key]['humidity'].append(item['main']['humidity'])
                daily_data[day_key]['wind'].append(item['wind']['speed'])
            
            # Calculate daily values
            for day_key, data in daily_data.items():
                if data['temps']:
                    daily_temps_max.append(max(data['temps']))
                    daily_temps_min.append(min(data['temps']))
                    daily_humidity.append(sum(data['humidity']) / len(data['humidity']))
                    daily_wind.append(sum(data['wind']) / len(data['wind']))
                    daily_dates.append(day_key)
    except Exception as e:
        print(f"Error processing daily forecast: {str(e)}")

    # Return statement med alle indsamlede data
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
