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

    # Check if hourly data exists before visualization
    if climate_data.get("hourly_temperatures") and climate_data.get("hourly_dates"):
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

    # Check if daily data exists before visualization
    if climate_data.get("daily_temps_max") and climate_data.get("daily_dates"):
        # Daily temperature range visualization
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

    # Check if hourly humidity and wind data exists
    if (climate_data.get("hourly_humidity") and climate_data.get("hourly_wind") and 
        climate_data.get("hourly_dates")):
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
        if climate_data.get("hourly_temperatures") and len(climate_data["hourly_temperatures"]) >= 8:
            avg_temp = sum(climate_data["hourly_temperatures"][:8]) / 8
            avg_humidity = sum(climate_data["hourly_humidity"][:8]) / 8
            st.metric("Avg. Temp (next 24h)", f"{avg_temp:.1f}°C")
            st.metric("Avg. Humidity (next 24h)", f"{avg_humidity:.1f}%")
