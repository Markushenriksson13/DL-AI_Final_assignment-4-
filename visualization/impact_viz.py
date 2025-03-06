import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from utils.constants import (TEMP_HIGH_THRESHOLD, TEMP_LOW_THRESHOLD, 
                            HUMIDITY_HIGH_THRESHOLD, HUMIDITY_LOW_THRESHOLD, 
                            WIND_HIGH_THRESHOLD, WIND_LOW_THRESHOLD,
                            SEVERE_NEGATIVE_IMPACT, NEGATIVE_IMPACT, 
                            POSITIVE_IMPACT)

def display_impact_data(impact_data, location, industry):
    """Display impact analysis visualizations."""
    if "error" in impact_data:
        st.error(f"Weather impact analysis: {impact_data['error']}")
        return
    
    st.subheader(f"Weather Impact Analysis for {industry} in {location}")
    
    # Display impact scores
    impacts = impact_data["impacts"]
    
    # Create impact visualization
    impact_types = ["Temperature", "Humidity", "Wind"]
    impact_scores = [
        impacts["temperature"]["impact_score"],
        impacts["humidity"]["impact_score"],
        impacts["wind"]["impact_score"]
    ]
    
    # Display overall impact i en separat boks
    overall = impact_data["overall_impact"]
    st.info(f"""
    **Overall Weather Impact on {industry} Sector:**
    
    Impact Score: {overall['score']:.1f}
    
    Interpretation: {overall['interpretation']}
    """)
    
    # Opret kolonner til impact information og visualisering
    col1, col2 = st.columns([2, 3])
    
    # Først viser vi impact details i venstre kolonne
    with col1:
        st.markdown("### Weather Impact Details")
        
        st.markdown("**Temperature Impact:**")
        st.markdown(f"- Score: {impacts['temperature']['impact_score']:.1f}")
        st.markdown(f"- {impacts['temperature']['description']}")
        
        st.markdown("**Humidity Impact:**")
        st.markdown(f"- Score: {impacts['humidity']['impact_score']:.1f}")
        st.markdown(f"- {impacts['humidity']['description']}")
        
        st.markdown("**Wind Impact:**")
        st.markdown(f"- Score: {impacts['wind']['impact_score']:.1f}")
        st.markdown(f"- {impacts['wind']['description']}")
        
        st.markdown("**Current Weather Condition Impact:**")
        st.markdown(f"- {impact_data['condition_impact']}")
    
    # Derefter viser vi visualiseringen i højre kolonne
    with col2:
        # Create impact score visualization
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = sns.barplot(x=impact_types, y=impact_scores, 
                          palette=['#ff9999' if x < 0 else '#99ff99' for x in impact_scores], ax=ax)
        
        # Add a horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add data labels with adjusted positions
        for i, bar in enumerate(bars.patches):
            height = bar.get_height()
            # Juster y-positionen baseret på om værdien er positiv eller negativ
            if height >= 0:
                y_pos = height + 0.3
                va = 'bottom'
            else:
                y_pos = height - 0.5
                va = 'top'
            
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos,
                f'{impact_scores[i]:.1f}',
                ha='center',
                va=va,
                fontsize=12
            )
        
        # Tilføj titel og labels med justeret størrelse og spacing
        ax.set_title(f'Weather Impact Scores for {industry} Sector', fontsize=14, pad=15)
        ax.set_ylabel('Impact Score (-10 to +10)', fontsize=12, labelpad=8)
        ax.set_ylim(-10, 10)
        
        # Juster x-akse labels
        plt.xticks(fontsize=12)
        ax.tick_params(axis='x', pad=8)
        plt.yticks(fontsize=12)
        
        # Øg padding for at sikre plads til titel og værdier
        plt.tight_layout()
        st.pyplot(fig)

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
        
        # Create generic recommendations for other sectors
        if industry not in sector_advice:
            sector_advice[industry] = {
                "high_temp": f"Adjust cooling systems for {industry} operations",
                "low_temp": f"Implement cold weather procedures for {industry}",
                "high_humidity": f"Monitor equipment and materials sensitive to humidity in {industry}",
                "low_humidity": f"Address dry conditions impact on {industry} operations",
                "high_wind": f"Secure equipment and materials from wind damage in {industry}",
                "low_wind": f"Optimal conditions for {industry} outdoor operations"
            }
        
        temp_dev = impact_data["current_weather"]["temperature"] - impact_data["average_weather"]["temperature"]
        humid_dev = impact_data["current_weather"]["humidity"] - impact_data["average_weather"]["humidity"]
        wind_dev = impact_data["current_weather"]["wind_speed"] - impact_data["average_weather"]["wind_speed"]
        
        recommendations = []
        advice = sector_advice[industry]
        
        if temp_dev > TEMP_HIGH_THRESHOLD:
            recommendations.append(advice["high_temp"])
        elif temp_dev < TEMP_LOW_THRESHOLD:
            recommendations.append(advice["low_temp"])
            
        if humid_dev > HUMIDITY_HIGH_THRESHOLD:
            recommendations.append(advice["high_humidity"])
        elif humid_dev < HUMIDITY_LOW_THRESHOLD:
            recommendations.append(advice["low_humidity"])
            
        if wind_dev > WIND_HIGH_THRESHOLD:
            recommendations.append(advice["high_wind"])
        elif wind_dev < WIND_LOW_THRESHOLD:
            recommendations.append(advice["low_wind"])
        
        if not recommendations:
            recommendations.append("Weather conditions are near normal - maintain standard operations.")
        
        return "\n".join([f"- {rec}" for rec in recommendations])
        
    except Exception as e:
        return f"Error generating recommendations: {str(e)}"

def get_context_guidance(industry, temp, wind, impact_score):
    """Provide context-specific guidance based on current conditions."""
    if impact_score < SEVERE_NEGATIVE_IMPACT:
        return f"⚠️ Critical weather conditions for {industry} sector. Consider implementing emergency measures."
    elif impact_score < NEGATIVE_IMPACT:
        return f"⚠️ Suboptimal conditions. Follow recommendations above to minimize impact."
    elif impact_score > POSITIVE_IMPACT:
        return f"✅ Optimal conditions for {industry} activities. Capitalize on favorable weather."
    else:
        return f"ℹ️ Normal operating conditions for {industry} sector. Maintain standard procedures."
