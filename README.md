# Climate & Sustainability Analysis Platform 🌍🌤️

A web application that provides climate data analysis and industry-specific weather impact assessments using real-time and historical data from OpenWeatherMap API.

## Features

- **Climate Data Analysis** 📊: View historical temperature, precipitation, humidity, and wind trends for any location
- **Industry-Specific Impact Assessment** 🏭: Analyze how weather patterns affect different sectors (Agriculture, Energy, Transportation, Tourism, Construction, Retail)
- **AI-Powered Recommendations** 🤖: Get tailored sustainability recommendations based on climate data and industry
- **Data Visualization** 📈: Interactive charts and graphs for easy data interpretation

## Project Structure
```
/
├── main.py                # Main application with Streamlit UI
├── utils/
│   ├── __init__.py
│   ├── constants.py       # Constants for threshold values
│   └── helpers.py         # Helper functions like handle_api_error
├── data/
│   ├── __init__.py
│   ├── climate_data.py    # Functions for retrieving climate data
│   └── impact_data.py     # Functions for impact analysis
├── visualization/
│   ├── __init__.py
│   ├── climate_viz.py     # climate data visualization
│   └── impact_viz.py      # impact data visualization
└── ai/
    ├── __init__.py
    ├── agents.py          # Agent-definitions
    ├── tasks.py           # Task-definitions
    ├── tools.py           # Tool definitions (analyze_trend etc.)
    └── crew.py            # Crew-setup & management
```
## API Requirements ⚙️

This application requires two API keys to function properly:

1. **OpenWeatherMap API Key** 🌦️
   - A Medium plan or higher is recommended for full historical data access (get if for free below)
   - Sign up for that free (only student/professor using university email) at [OpenWeatherMap](https://docs.openweather.co.uk/our-initiatives/student-initiative)

3. **Together AI API Key** 🧠
   - Sign up at [Together AI](https://www.together.ai/)
   - 5$ Credit for free is given at signup
   - We utilize the "Free-versions" of the models:
      - Llama-3.3-70B-Instruct-Turbo-Free & DeepSeek-R1-Distill-Llama-70B-Free
   - Required for the AI analysis and recommendation features

### Setting Up API Keys

1. Create a `.env` file in the root directory of the project
2. Add your API keys in the following format:
   ```
   OPENWEATHER_API_KEY=your_openweather_api_key_here
   TOGETHER_API_KEY=your_together_api_key_here
   ```
3. Save the file and restart the application if it's already running

⚠️ **Note**: Without these API keys, the application will not be able to retrieve weather data or generate AI recommendations.

## AI Agents and Analysis System 🧪

The platform utilizes a sophisticated multi-agent system powered by CrewAI and LangChain:

### Agents

1. **Climate Data Analyst** 🌡️
   - Role: Analyzes weather patterns and their impacts
   - Tools: 
     - Climate data retrieval
     - Trend analysis
   - Model: DeepSeek-R1-Distill-Llama-70B-Free
   - Primary focus: Temperature trends, precipitation patterns, and extreme weather risks

2. **Impact Assessment Specialist** 📉
   - Role: Evaluates weather impacts on specific industries
   - Tools:
     - Weather impact analysis
     - Trend analysis
   - Model: Llama-3.3-70B-Instruct-Turbo-Free
   - Primary focus: Operational impacts and resource efficiency

3. **Weather Advisory Specialist** 💼
   - Role: Provides actionable recommendations
   - Model: DeepSeek-R1-Distill-Llama-70B-Free
   - Primary focus: Risk mitigation and adaptation strategies

### Analysis Process ⏱️

The analysis is performed in three sequential stages:

1. **Climate Analysis** 
   - Historical data analysis
   - Pattern identification
   - Anomaly detection

2. **Impact Assessment** 
   - Industry-specific evaluation
   - Resource efficiency analysis
   - Safety considerations
   - Economic implications

3. **Recommendations**
   - Short-term adjustments
   - Medium-term strategies
   - Long-term resilience planning
   - Risk mitigation steps

### Tools and Capabilities 🛠️

- **Climate Data Tool**: Retrieves comprehensive climate data using OpenWeatherMap API
- **Impact Analysis Tool**: Evaluates sector-specific weather impacts
- **Trend Analysis Tool**: Analyzes patterns and changes in weather metrics

## Models Used 🧠

The platform leverages several AI models from Together AI:
- DeepSeek-R1-Distill-Llama-70B: Used for climate analysis and recommendations
- Llama-3.3-70B-Instruct-Turbo: Specialized for impact assessment

## Setting Up the Python 3.10 Virtual Environment 🐍

Follow the steps below to create and activate a virtual environment and install the required Python dependencies.

### Install Python 3.10: 
#### Installing Python 3.10 on macOS (With Homebrew Option)

#### MacOS: Install Homebrew (If Needed)
If you don't already have Homebrew installed, you can install it by running the following command in your terminal:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**Download Python 3.10 on MacOS using Homebrew:**
```bash
brew install python@3.10
```

#### On Windows

1. Download the Python 3.10.6 installer from the [official Python website](https://www.python.org/downloads/release/python-31016/).
2. Run the installer and follow the installation steps. Make sure to check the option "Add Python 3.10 to PATH" during installation.


### 1. Create a Virtual Environment

Run the following command to create a virtual environment using Python 3.10:

```bash
python3.10 -m venv venv
```
This will create a folder named venv in the project root.

### 2. Activate the Virtual Environment
Activate the virtual environment with the following command:

#### On Linux/macOS:
```bash
source venv/bin/activate
```
#### On Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies
With the virtual environment active, install the project's dependencies from the requirements.txt file using:

```bash
pip install -r requirements.txt
```

## Usage 🚀

Open terminal window from project folder
Run the application with Streamlit:
```
streamlit run main.py
```

Then enter a location, select an industry sector, and add any specific environmental concerns to receive a comprehensive analysis.

## Data Sources

This application uses the OpenWeatherMap API for data collection:
- Current weather data
- Statistical weather data (monthly and yearly aggregations)
- Historical weather data
- Geocoding services

## Requirements

- Python 3.10+
- OpenWeatherMap API key (Medium plan or higher for full historical data access)
- Together API key for AI recommendations
