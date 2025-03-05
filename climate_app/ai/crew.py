import streamlit as st
from crewai import Task, Crew, Process
from ai.agents import create_agents

def create_tasks(location, industry, specific_concerns):
    """Create AI-powered analysis and recommendations."""
    try:
        # Opret simpel progress bar og status tekst
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Vis startbesked
        progress_bar.progress(25)
        status_text.text("Analysis in progress... (this may take several minutes)")
        
        # Få agenter
        climate_analyst, impact_analyst, recommendation_specialist = create_agents()
        
        # Definer tasks
        climate_analysis = Task(
            description=f"""Analyze the climate data for {location} and identify key patterns 
            and trends that could affect the {industry} sector. Focus on:
            1. Temperature trends and anomalies
            2. Precipitation patterns
            3. Wind conditions
            4. Extreme weather risks
            
            Consider the specific concerns: {specific_concerns}""",
            expected_output="""Detailed analysis of climate patterns including:
            - Temperature trend analysis
            - Precipitation analysis
            - Wind pattern analysis
            - Identification of extreme weather risks
            - Impact assessment for the specified industry""",
            agent=climate_analyst,
            task_name="Climate Analysis"
        )
        
        progress_bar.progress(50)
        
        impact_assessment = Task(
            description=f"""Evaluate how the current and forecasted weather conditions will 
            impact the {industry} sector in {location}. Consider:
            1. Operational impacts
            2. Resource efficiency
            3. Safety considerations
            4. Economic implications
            
            Address these specific concerns: {specific_concerns}""",
            expected_output="""Comprehensive impact assessment including:
            - Detailed operational impact analysis
            - Resource efficiency evaluation
            - Safety risk assessment
            - Economic impact projections
            - Specific responses to stated concerns""",
            agent=impact_analyst,
            context=[climate_analysis],
            task_name="Impact Assessment"
        )
        
        recommendations = Task(
            description=f"""Based on the climate analysis and impact assessment, provide 
            specific, actionable recommendations for the {industry} sector in {location}. 
            Include:
            1. Short-term operational adjustments
            2. Medium-term adaptation strategies
            3. Long-term resilience measures
            4. Risk mitigation steps
            
            Ensure recommendations address: {specific_concerns}""",
            expected_output="""Actionable recommendations including:
            - Specific short-term adjustments
            - Medium-term adaptation strategies
            - Long-term resilience planning
            - Detailed risk mitigation steps
            - Timeline for implementation""",
            agent=recommendation_specialist,
            context=[impact_assessment],
            task_name="Recommendations"
        )
        
        # Opret og kør crew
        try:
            crew = Crew(
                agents=[climate_analyst, impact_analyst, recommendation_specialist],
                tasks=[climate_analysis, impact_assessment, recommendations],
                verbose=True,
                process=Process.sequential,
                max_retries=2
            )
            
            # Kør analysen
            result = crew.kickoff()
            
            # Opdater progress når færdig
            progress_bar.progress(100)
            status_text.text("Analysis completed!")
            
            return result
            
        except Exception as e:
            st.error(f"Error in AI Analysis: {str(e)}")
            progress_bar.progress(100)
            status_text.text("Analysis failed")
            return None
            
    except Exception as e:
        st.error(f"Error setting up analysis: {str(e)}")
        return None
