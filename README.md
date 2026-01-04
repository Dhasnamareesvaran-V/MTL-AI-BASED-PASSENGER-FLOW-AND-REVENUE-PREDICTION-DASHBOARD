METRO TRANSPORTATION LIMITED (MTL) - AI BASED PASSENGER FLOW PREDICTION DASHBOARD

OVERVIEW

MTL Passenger Flow Prediction Dashboard is an AI‑powered application built during an internship at Bigsibucks Innovation Pvt Ltd. It integrates Computer Vision (CV) and Time Series Forecasting (Prophet) to automate passenger demographic analysis and revenue forecasting. The system predicts age and gender from uploaded passenger images, aggregates statistics across multiple passengers, and forecasts passenger flow and revenue trends. The project emphasizes modular design, reproducible setup, and a polished Streamlit UI tailored for real‑world transportation analytics.

PROBLEM STATEMENT

Metro operators and planners often struggle with manual data collection and forecasting, which is time‑consuming, error‑prone, and lacks transparency. There is a need for an automated system that can:
Analyze passenger demographics from images.
Forecast passenger flow and revenue using historical data.
Provide clear, explainable visualizations for decision‑making.

OBJECTIVES OF THE PROJECT

Automate passenger demographic prediction (age & gender) from uploaded images.  
Support multiple image uploads (e.g., 50 passengers) with aggregate statistics.  
Forecast passenger flow and revenue using Prophet model outputs.  
Provide transparent insights with bar charts, pie charts, and line charts.  
Deliver a polished, recruiter‑friendly UI with branding and professional presentation.

KEY FEATURES

Automated CV Predictions: Age and gender classification using trained Keras models.  
Aggregate Analysis: Average age, gender distribution, and most frequent age group across multiple passengers.  
Forecasting: Prophet‑based revenue predictions with passenger flow estimation.  
Transparent Visualizations:  
1.Bar chart (Flow vs Revenue with labels).  
2.Pie chart (Gender distribution).  
3.Age group frequency chart (which age group uses the station most).  
4.Line charts for revenue and passenger flow trends.  
Polished UI/UX: Clean Streamlit interface with branding, logo, and animated gradient background.

TECH STACK

Language: Python  
Frontend: Streamlit

Core Libraries:
1.OpenCV: Image preprocessing.  
2.Keras/TensorFlow: Age & gender prediction models.  
3.Prophet: Revenue forecasting.  
4.Pandas & NumPy: Data handling.  
5.Matplotlib: Visualizations.

Environment: Python 3.9+ with virtual environment (venv).  
Version Control: Git + GitHub for code, docs, and issue tracking.

PACKAGES

1. streamlit
2. opencv-python
3. tensorflow / keras
4. pandas
5. numpy
6. matplotlib
7. prophet

ARCHITECTURE AND WORKFLOW

Design principles: Modular architecture, clear separation of concerns (UI, CV models, forecasting), and reproducibility.

Conceptual flow:

[User Uploads Passenger Images]
↓
[CV Models: Age & Gender Prediction]
↓
[Aggregator: Average Age, Gender Distribution, Age Group Frequency]
↓
[Forecasting Module: Prophet Revenue & Flow Prediction]
↓
[Streamlit UI Output: Charts + Stats]

PROJECT STRUCTURE

INSTALLATION

Prerequisites: Python 3.9+, Git, and a working internet connection.

Create and activate a virtual environment:

Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1

mac OS/Linux
python3 -m venv venv
source venv/bin/activate

INSTALL DEPENDENCIES

pip install --upgrade pip
pip install -r requirements.txt

USAGE

Prepare inputs:
Passenger images: Upload .jpg/.png/.jpeg files via the UI.

Forecast data: Ensure forecast_revenue_prophet.csv is in models/forecasting/.

Run the app:
streamlit run scripts/app.py

WORKFLOW IN THE UI:

Upload passenger images (single or multiple).
App predicts age & gender, aggregates stats, and forecasts flow & revenue.
Review results: Average age, gender distribution, most frequent age group, bar/pie charts, and forecast trends.

CONFIGURATION AND CUSTOMIZATION

Average Fare: Adjust fare assumption in app.py for passenger flow calculation.
Model Selection: Swap or fine‑tune CV models in models/cv/.
Forecast Data: Replace Prophet CSV with updated forecasts in models/forecasting/.
UI Branding: Update logo and color scheme in data/assets/ and app.py.

MODULES OVERVIEW

Age/Gender Models: Predict passenger demographics from images.
Aggregator: Computes average age, gender counts, and age group frequency.
Forecasting Module: Uses Prophet CSV to forecast revenue and flow.
UI (Streamlit): Displays stats, charts, and branding.

SAMPLE DATA

The project includes example forecast data and assets for quick testing:
Forecast CSV: forecast_revenue_prophet.csv in models/forecasting/.
Assets: Logo (logo.jpg) in data/assets/.
Passenger Images: Upload your own .jpg/.png/.jpeg files for testing.

FUTURE ENHANCEMENTS

Crowd Image Support: Add face detection to handle group photos.
Export Features: Allow recruiters to download aggregate stats as CSV/Excel.
Real‑Time Data Integration: Connect to live metro feeds for dynamic forecasting.
Cloud Deployment: Host on Streamlit Cloud, Azure, or Hugging Face Spaces.
Explainability Features: Show why a passenger was classified into a certain age group.
