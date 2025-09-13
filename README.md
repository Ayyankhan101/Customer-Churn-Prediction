# Customer Churn Prediction

This project predicts customer churn using a machine learning model. It includes a Streamlit web application for interactive analysis and a Jupyter notebook for the model development process.

## Features

- **Interactive Dashboard:** A Streamlit application to visualize churn data, filter customers, and see model predictions.
- **Churn Prediction Model:** A Random Forest Classifier trained to predict customer churn.
- **Exploratory Data Analysis:** A detailed Jupyter notebook covering data analysis, feature engineering, and model training.

## Getting Started

### Prerequisites

- Python 3.7+
- Pip

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```
   This will open the interactive dashboard in your web browser.

2. **Explore the analysis in the Jupyter notebook:**
   ```bash
   jupyter notebook Churn_Analysis.ipynb
   ```

## Dataset

The project uses two datasets:
- `customer_churn_dataset-training-master.csv`: For training the model.
- `customer_churn_dataset-testing-master.csv`: For testing the model.

The datasets include the following columns:
- `CustomerID`
- `Age`
- `Gender`
- `Tenure`
- `Usage Frequency`
- `Support Calls`
- `Payment Delay`
- `Subscription Type`
- `Contract Length`
- `Total Spend`
- `Last Interaction`
- `Churn`

## Model

The churn prediction model is a `RandomForestClassifier` from the scikit-learn library. The trained model is saved in the `churn_model.pkl` file.

The model development process, including data preprocessing, feature engineering, and model evaluation, is detailed in the `Churn_Analysis.ipynb` notebook.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
