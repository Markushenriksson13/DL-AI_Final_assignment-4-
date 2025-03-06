from utils.helpers import extract_values

def analyze_trend(data: dict, metric: str) -> dict:
    """Analyze trends in provided weather data metrics."""
    try:
        # convert string to dict if necessary
        if isinstance(data, str):
            try:
                import json
                data = json.loads(data)
            except json.JSONDecodeError:
                return {"error": "Invalid JSON format"}
        
        # find values using a single helper function
        values = extract_values(data, metric)
        
        if values is None:
            return {"error": f"Metric {metric} not found in data"}
        if not isinstance(values, list) or not values:
            return {"error": "No valid data points found"}
            
        # calculate statistics
        avg = sum(values) / len(values)
        trend = "increasing" if values[-1] > values[0] else "decreasing"
        change = ((values[-1] - values[0]) / values[0]) * 100 if values[0] != 0 else 0
        
        return {
            "average": round(avg, 2),
            "trend": trend,
            "change_percent": round(change, 2)
        }
    except Exception as e:
        return {"error": f"Error analyzing trend: {str(e)}"}
