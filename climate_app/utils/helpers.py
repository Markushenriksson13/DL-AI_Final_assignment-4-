def handle_api_error(error_msg, exception=None):
    """Centraliseret fejlhåndtering for API-kald"""
    if exception:
        error_details = f": {str(exception)}"
    else:
        error_details = ""
    return {"error": f"{error_msg}{error_details}"}
    
def extract_values(data, metric):
    """Helper to extract values from different data structures"""
    if not isinstance(data, dict):
        return None
        
    if metric == "all" and "data" in data and isinstance(data["data"], dict):
        results = {}
        for key, val in data["data"].items():
            if isinstance(val, list) and val:
                results[key] = {
                    "average": round(sum(val) / len(val), 2),
                    "trend": "increasing" if val[-1] > val[0] else "decreasing", 
                    "change_percent": round(((val[-1] - val[0]) / val[0]) * 100, 2) if val[0] != 0 else 0
                }
        return results
    
    # try to find values in different data locations
    if "data" in data and isinstance(data["data"], dict) and metric in data["data"]:
        return data["data"][metric]
    elif metric in data:
        return data[metric]
    
    return None
