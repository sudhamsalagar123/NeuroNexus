# 🚢 Titanic Survival Prediction

![Titanic](https://img.shields.io/badge/Dataset-Titanic-blue)
![Python](https://img.shields.io/badge/Python-3.7+-brightgreen)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange)
![Gradio](https://img.shields.io/badge/Gradio-3.0+-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

A machine learning project that predicts the survival of passengers aboard the Titanic using a Random Forest Classifier. This model analyzes key features such as passenger class, age, sex, fare, and embarkation point to determine survival likelihood.

## 📊 Overview

- **Dataset**: Titanic dataset (train.csv, test.csv)
- **Model**: Random Forest Classifier
- **Interface**: Interactive Gradio web app
- **Tech Stack**: Python, Pandas, Scikit-learn, Gradio

## 🚀 Features

- ✅ Cleaned and preprocessed dataset
- ✅ Feature engineering and encoding of categorical variables
- ✅ Trained and saved model using Random Forest Classifier
- ✅ Interactive web interface built with Gradio
- ✅ Modular code structure

## 🛠️ Installation

```bash
# Clone this repository
git clone https://github.com/sudham-salagar/titanic-survival-prediction.git
cd titanic-survival-prediction

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

## 📋 Usage

### Training the Model

```bash
python train_model.py
```

### Running the Web Interface

```bash
python app.py
```

Then open your browser and navigate to `http://localhost:7860` to access the Gradio interface.

## 🧠 Model Details

- **Algorithm**: Random Forest Classifier
- **Target Variable**: Survived (0 = Did not survive, 1 = Survived)
- **Input Features**:
  - `Pclass` – Ticket class (1, 2, 3)
  - `Sex` – Gender (male, female)
  - `Age` – Passenger age
  - `SibSp` – Siblings/Spouses aboard
  - `Parch` – Parents/Children aboard
  - `Fare` – Ticket fare
  - `Embarked` – Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 0.82 |
| Precision | 0.78 |
| Recall | 0.71 |
| F1 Score | 0.74 |

## 📦 Project Structure

```
titanic-survival-prediction/
├── data/
│   ├── train.csv
│   └── test.csv
├── models/
│   └── random_forest_model.pkl
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_model_training.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model.py
│   └── utils.py
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
```

## 💻 Example Code

```python
# Sample code to make predictions
from src.model import load_model
from src.data_preprocessing import preprocess_data
import pandas as pd

# Load model
model = load_model('models/random_forest_model.pkl')

# Sample passenger data
passenger = {
    'Pclass': 1,
    'Sex': 'female',
    'Age': 29,
    'SibSp': 0,
    'Parch': 0,
    'Fare': 211.3375,
    'Embarked': 'S'
}

# Convert to DataFrame and preprocess
df = pd.DataFrame([passenger])
processed_data = preprocess_data(df)

# Predict survival
prediction = model.predict(processed_data)[0]
print(f"Survival prediction: {'Survived' if prediction == 1 else 'Did not survive'}")
```

## 📸 Screenshots

### Web Interface
![Image](https://github.com/user-attachments/assets/d36a5a16-126e-4c81-ad53-ee3e6e543c7a)


### Prediction Example

![Image](https://github.com/user-attachments/assets/21f40426-29bd-44b7-8c40-085e1f8a2461)

## 📚 Acknowledgements

- [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic)
- [Gradio](https://www.gradio.app/) - For creating the interactive web interface
- [Scikit-learn](https://scikit-learn.org/) - For machine learning tools in Python

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙌 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## ✨ Author

Created by Sudham Salagar

## 📧 Contact

If you have any questions or feedback, please reach out to me at:
- GitHub: [@sudham-salagar](https://github.com/sudhamsalagar123)
