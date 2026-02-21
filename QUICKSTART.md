# Quick Start Guide

## Installation (5 minutes)

1. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set up environment** (optional, for AI recommendations):
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key

3. **Run the dashboard**:
```bash
streamlit run app/app.py
```

## First Run

1. **Upload Sample Data**:
   - Go to "Data Upload" page
   - The sample file `data/raw/user_data.csv` is already included
   - Or upload your own CSV file

2. **Select Target Column**:
   - Choose which column you want to predict
   - Example: `investment_success` in the sample data

3. **Explore Data**:
   - Go to "EDA Dashboard"
   - View statistics and visualizations

4. **Train Model**:
   - Go to "Model Training"
   - Click "Train Model"
   - View performance metrics

5. **Make Predictions**:
   - Go to "Predictions"
   - Enter feature values or upload CSV
   - Get predictions with risk levels

6. **Get Explanations**:
   - Go to "Explanations"
   - See WHY the model made its prediction
   - View feature contributions

7. **Get Recommendations**:
   - Go to "Recommendations"
   - Select domain context
   - Get AI-powered advice

## Sample Data Format

Your CSV should have:
- Feature columns (numerical or categorical)
- One target column (what you want to predict)

Example:
```csv
age,income,savings,target
25,50000,10000,0
30,75000,25000,1
```

## Troubleshooting

**Import errors?**
- Make sure you're in the project root directory
- Install requirements: `pip install -r requirements.txt`

**SHAP errors?**
- Try: `pip install shap --no-cache-dir`

**Database errors?**
- The database will be created automatically
- Check that `database/` folder exists

**OpenAI API errors?**
- The system works without API key (uses fallback recommendations)
- Add your key to `.env` file for AI recommendations

## Next Steps

- Try different models in "Model Training"
- Experiment with your own data
- Explore the Jupyter notebooks in `notebooks/`
- Customize the code for your use case
