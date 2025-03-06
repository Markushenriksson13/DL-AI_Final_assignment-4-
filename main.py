import streamlit as st
import os
from dotenv import load_dotenv
from data.climate_data import get_climate_data
from data.impact_data import get_weather_impact_analysis
from ai.crew import create_tasks
from visualization.climate_viz import display_climate_data
from visualization.impact_viz import display_impact_data, get_sector_recommendations, get_context_guidance

# Load environment variables
load_dotenv()

def main():
    st.set_page_config(
        page_title="Climate & Sustainability Analysis Platform",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("🌎 Climate & Sustainability Analysis Platform")
    st.write("Analyze climate trends and sustainability metrics for your location and industry.")
    
    # User inputs
    location = st.text_input("Enter location (city, country):", "Copenhagen, Denmark")
    industry = st.selectbox("Select industry sector:", 
                           ["Agriculture", "Energy", "Transportation", "Tourism", "Construction", "Retail"])
    concerns = st.text_area("Any specific environmental concerns?", 
                           "How will the anticipated weather changes impact our operations over the coming year?")
    
    if st.button("Analyze"):
        with st.spinner("Analyzing weather and climate data..."):
            try:
                # call functions directly
                climate_data = get_climate_data(location)
                impact_data = get_weather_impact_analysis(location, industry)
                
                # display climate data visualizations
                display_climate_data(climate_data, location)
                
                # display impact data visualizations
                display_impact_data(impact_data, location, industry)
                
                # create and display AI analysis
                st.subheader("AI Analysis & Recommendations")
                st.write("Performing detailed AI analysis of climate patterns and impacts...")
                
                # generate AI analysis
                try:
                    tasks = create_tasks(location, industry, concerns)
                    
                    if tasks is None:
                        # provide alternative quick recommendations when AI analysis fails
                        st.warning("AI analysis couldn't be completed. Here are some basic recommendations:")
                        if "error" not in impact_data and "overall_impact" in impact_data:
                            recommendations = get_sector_recommendations(industry, impact_data)
                            st.markdown(f"### Quick Recommendations for {industry} in {location}")
                            st.markdown(recommendations)
                            
                            overall_score = impact_data["overall_impact"]["score"]
                            guidance = get_context_guidance(industry, 
                                                          impact_data["current_weather"]["temperature"],
                                                          impact_data["current_weather"]["wind_speed"],
                                                          overall_score)
                            st.info(guidance)
                    else:
                        # display results when AI analysis is successful
                        st.markdown("### AI Analysis Results")
                        st.markdown(tasks)  # this displays the raw result
                        
                        # split results into sections for better readability
                        if isinstance(tasks, str) and len(tasks) > 0:
                            sections = tasks.split("**")
                            if len(sections) > 1:
                                for i in range(1, len(sections)):
                                    section = sections[i]
                                    if i < len(sections) - 1:
                                        section = section + "**"  # restore the removed "**"
                                    st.markdown(f"**{section}")
                except Exception as e:
                    st.error(f"Error in AI analysis: {str(e)}")
            except Exception as e:
                st.error(f"Error in data analysis: {str(e)}")

if __name__ == "__main__":
    main()
