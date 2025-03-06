import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

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
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 14,
        'figure.titlesize': 20
    })
    
    # Visualizations
    st.header(f"Climate Trends for {location}")
    st.subheader("Statistical Data Analysis")
    
    # Display additional statistical data if available
    if "statistics" in climate_data and climate_data["statistics"]:
        stats = climate_data["statistics"]
        
        with st.expander("📊 Detailed Statistical Data (Current Month)", expanded=True):
            sunshine_hours = f"### ☀️ Sunshine Hours\n* **{stats['sunshine_hours']} hours/month**" if 'sunshine_hours' in stats else ""
            
            st.markdown(f"""
# Statistical Climate Data for {location}

### 🌡️ Temperature Records
* Record Low: **{stats['temperature']['record_min']}°C**
* Record High: **{stats['temperature']['record_max']}°C**
* Average Low: **{stats['temperature']['average_min']}°C**
* Average High: **{stats['temperature']['average_max']}°C**

### 💧 Humidity
* Average: **{stats['humidity']['mean']}%**
* Range: **{stats['humidity']['min']}% - {stats['humidity']['max']}%**

### 🌪️ Wind Speed
* Average: **{stats['wind']['mean']} m/s**
* Range: **{stats['wind']['min']} - {stats['wind']['max']} m/s**

### 🌧️ Precipitation
* Average: **{stats['precipitation']['mean']} mm/day**
* Maximum: **{stats['precipitation']['max']} mm/day**

{sunshine_hours}

*Note: This statistical data is calculated based on historical measurements.*
""")
    
    # Monthly Trends Section
    st.subheader("Monthly Climate Trends")
    col1, col2 = st.columns(2)
    
    with col1:
        # Temperature plot
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.lineplot(x=range(len(months)), y=temp_data, marker='o', linewidth=3, color='#1f77b4', ax=ax)
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(months, rotation=45, ha='right')
        ax.set_xlabel('Month', fontsize=16, labelpad=10)
        ax.set_ylabel('Temperature (°C)', fontsize=16, labelpad=10)
        ax.set_title('Temperature Trends', fontsize=18, pad=20)
        plt.tight_layout()
        for i, (x, y) in enumerate(zip(range(len(months)), temp_data)):
            ax.text(x, y + 0.5, f'{y:.1f}°C', ha='center', va='bottom', fontsize=12)
        st.pyplot(fig)
    
    with col2:
        # Precipitation plot
        fig, ax = plt.subplots(figsize=(12, 7))
        bars = sns.barplot(x=range(len(months)), y=precip_data, color='#2ca02c', ax=ax)
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(months, rotation=45, ha='right')
        ax.set_xlabel('Month', fontsize=16, labelpad=10)
        ax.set_ylabel('Precipitation (mm/month)', fontsize=16, labelpad=10)
        ax.set_title('Precipitation Trends', fontsize=18, pad=20)
        plt.tight_layout()
        for i, (x, y) in enumerate(zip(range(len(months)), precip_data)):
            ax.text(x, y + 0.5, f'{y:.1f}mm', ha='center', va='bottom', fontsize=12)
        st.pyplot(fig)

    # Humidity and Wind Section
    st.subheader("Humidity and Wind Patterns")
    col1, col2 = st.columns(2)
    
    with col1:
        # Humidity plot
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.lineplot(x=range(len(months)), y=humidity_data, marker='s', linewidth=3, color='#9467bd', ax=ax)
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(months, rotation=45, ha='right')
        ax.set_xlabel('Month', fontsize=16, labelpad=10)
        ax.set_ylabel('Humidity (%)', fontsize=16, labelpad=10)
        ax.set_title('Humidity Trends', fontsize=18, pad=20)
        plt.tight_layout()
        for i, (x, y) in enumerate(zip(range(len(months)), humidity_data)):
            ax.text(x, y + 2, f'{y:.0f}%', ha='center', va='bottom', fontsize=12)
        st.pyplot(fig)
    
    with col2:
        # Wind plot
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.lineplot(x=range(len(months)), y=wind_data, marker='d', linewidth=3, color='#d62728', ax=ax)
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(months, rotation=45, ha='right')
        ax.set_xlabel('Month', fontsize=16, labelpad=10)
        ax.set_ylabel('Wind Speed (m/s)', fontsize=16, labelpad=10)
        ax.set_title('Wind Speed Trends', fontsize=18, pad=20)
        plt.tight_layout()
        for i, (x, y) in enumerate(zip(range(len(months)), wind_data)):
            ax.text(x, y + 0.2, f'{y:.1f}m/s', ha='center', va='bottom', fontsize=12)
        st.pyplot(fig)

    # Detailed Forecasts Section
    st.subheader("Detailed Weather Forecasts")
    
    # Check if hourly data exists before visualization
    if climate_data.get("hourly_temperatures") and climate_data.get("hourly_dates"):
        col1, col2 = st.columns(2)
        with col1:
            with st.expander("🌡️ Detailed Temperature Forecast (3-Hour Intervals)", expanded=True):
                # Detailed temperature trend (3-hourly)
                fig, ax = plt.subplots(figsize=(12, 7))
                ax.plot(climate_data["hourly_dates"], climate_data["hourly_temperatures"], 
                        marker='o', label='Temperature (°C)', color='red', alpha=0.6, markersize=4)
                ax.fill_between(climate_data["hourly_dates"], 
                                [t-1 for t in climate_data["hourly_temperatures"]], 
                                [t+1 for t in climate_data["hourly_temperatures"]], 
                                color='red', alpha=0.2)
                ax.set_xlabel('Time', fontsize=16, labelpad=10)
                ax.set_ylabel('Temperature (°C)', fontsize=16, labelpad=10)
                ax.set_title('Detailed Temperature Forecast', fontsize=18, pad=20)
                
                # Optimize x-axis labels for better readability
                num_ticks = min(6, len(climate_data["hourly_dates"]))
                step = len(climate_data["hourly_dates"]) // num_ticks
                plt.xticks(range(0, len(climate_data["hourly_dates"]), step), 
                          [climate_data["hourly_dates"][i] for i in range(0, len(climate_data["hourly_dates"]), step)],
                          rotation=45, ha='right', fontsize=12)
                plt.yticks(fontsize=12)
                
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=14, loc='upper right')
                plt.tight_layout()
                st.pyplot(fig)

    # Check if daily data exists before visualization
    if climate_data.get("daily_temps_max") and climate_data.get("daily_dates"):
        with col2:
            with st.expander("📅 Daily Temperature Range Forecast", expanded=True):
                # Daily temperature range visualization
                fig, ax = plt.subplots(figsize=(12, 7))
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
                        'r--', label='Max Temperature', linewidth=2)
                ax.plot(dates_display, climate_data["daily_temps_min"], 
                        'b--', label='Min Temperature', linewidth=2)
                ax.set_xlabel('Date', fontsize=16, labelpad=10)
                ax.set_ylabel('Temperature (°C)', fontsize=16, labelpad=10)
                ax.set_title('Daily Temperature Range', fontsize=18, pad=20)
                plt.xticks(rotation=45, ha='right', fontsize=12)
                plt.yticks(fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=14, loc='upper right')
                plt.tight_layout()
                st.pyplot(fig)

    # Check if hourly humidity and wind data exists
    if (climate_data.get("hourly_humidity") and climate_data.get("hourly_wind") and 
        climate_data.get("hourly_dates")):
        col1, col2 = st.columns([3, 2])
        
        # Først viser vi visualiseringen i venstre kolonne
        with col1:
            with st.expander("🌪️ Humidity and Wind Speed Correlation", expanded=True):
                # Humidity and wind correlation
                fig, ax1 = plt.subplots(figsize=(8, 5))
                
                # Format x-axis for better readability
                num_ticks = min(5, len(climate_data["hourly_dates"]))
                step = max(len(climate_data["hourly_dates"]) // num_ticks, 1)
                x_ticks = range(0, len(climate_data["hourly_dates"]), step)
                x_labels = [climate_data["hourly_dates"][i] for i in x_ticks]
                
                # Plot humidity
                ax1.set_xlabel('Time', fontsize=12, labelpad=8)
                ax1.set_ylabel('Humidity (%)', fontsize=12, labelpad=8, color='blue')
                humidity_line = ax1.plot(climate_data["hourly_dates"], climate_data["hourly_humidity"], 
                                       color='blue', label='Humidity', linewidth=2)
                ax1.tick_params(axis='y', labelcolor='blue', labelsize=10)
                
                # Plot wind speed on secondary y-axis
                ax2 = ax1.twinx()
                ax2.set_ylabel('Wind Speed (m/s)', fontsize=12, labelpad=8, color='green')
                wind_line = ax2.plot(climate_data["hourly_dates"], climate_data["hourly_wind"], 
                                    color='green', label='Wind Speed', linewidth=2)
                ax2.tick_params(axis='y', labelcolor='green', labelsize=10)
                
                # Set x-axis ticks med mere plads og mindre tekst
                plt.xticks(x_ticks, x_labels, rotation=45, ha='right', fontsize=9)
                
                # Juster figur størrelse og margins
                plt.subplots_adjust(bottom=0.3)
                
                # Add title
                ax1.set_title('Humidity and Wind Speed Correlation', fontsize=14, pad=15)
                
                # Add legend
                lines = humidity_line + wind_line
                labels = ['Humidity', 'Wind Speed']
                ax1.legend(lines, labels, loc='upper right', fontsize=10)
                
                plt.tight_layout()
                st.pyplot(fig)
        
        # Derefter viser vi statistikken i højre kolonne
        with col2:
            # Tilføj forklarende tekst eller statistik
            if climate_data.get("hourly_humidity") and len(climate_data["hourly_humidity"]) > 0:
                avg_humidity = sum(climate_data["hourly_humidity"]) / len(climate_data["hourly_humidity"])
                max_humidity = max(climate_data["hourly_humidity"])
                min_humidity = min(climate_data["hourly_humidity"])
                
                avg_wind = sum(climate_data["hourly_wind"]) / len(climate_data["hourly_wind"])
                max_wind = max(climate_data["hourly_wind"])
                min_wind = min(climate_data["hourly_wind"])
                
                st.info(f"""
                **Humidity & Wind Statistics:**
                
                **Humidity:**
                - Average: {avg_humidity:.1f}%
                - Range: {min_humidity:.1f}% - {max_humidity:.1f}%
                
                **Wind Speed:**
                - Average: {avg_wind:.1f} m/s
                - Range: {min_wind:.1f} - {max_wind:.1f} m/s
                
                *These values represent the forecast period shown in the chart.*
                """)
            else:
                st.info("Detailed humidity and wind statistics not available for this location.")

    # Current Weather Overview
    st.subheader("Current Weather Overview")
    
    # Tilføj Current Weather Conditions hvis de er tilgængelige
    if climate_data.get("current_weather_condition"):
        st.info(f"""
        **Current Weather Conditions in {location}:**
        - Temperature: {climate_data.get('current_temperature', 'N/A')}°C
        - Humidity: {climate_data.get('current_humidity', 'N/A')}%
        - Wind Speed: {climate_data.get('current_wind', 'N/A')} m/s
        - Condition: {climate_data.get('current_weather_condition', 'N/A')}
        """)
    
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🌡️ Feels Like", 
                      f"{climate_data['current_feels_like']}°C" 
                      if climate_data['current_feels_like'] != 'N/A' else 'N/A')
        
        with col2:
            st.metric("🌪️ Air Pressure", 
                      f"{climate_data['current_pressure']} hPa" 
                      if climate_data['current_pressure'] != 'N/A' else 'N/A')
        
        # Add 24h forecasts if available
        if climate_data.get("hourly_temperatures") and len(climate_data["hourly_temperatures"]) >= 8:
            avg_temp = sum(climate_data["hourly_temperatures"][:8]) / 8
            avg_humidity = sum(climate_data["hourly_humidity"][:8]) / 8
            
            with col3:
                st.metric("🌡️ Avg. Temp (next 24h)", f"{avg_temp:.1f}°C")
            
            with col4:
                st.metric("💧 Avg. Humidity (next 24h)", f"{avg_humidity:.1f}%")
