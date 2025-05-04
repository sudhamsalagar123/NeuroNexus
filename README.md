# 🚢 Titanic Survival Prediction

A Titanic dataset to build a model that predicts whether a passenger on the Titanic survived or not. This is a classic beginner project with readily available data.
The dataset typically used for this project contains information about individual passengers, such as their age, gender, ticket class, fare, cabin, and whether or not they survived and that machine learning project that predicts whether a passenger survived the Titanic disaster using logistic regression. Built using Python and scikit-learn.

---

## 📁 Dataset

The dataset used in this project is the famous [Titanic dataset](https://www.kaggle.com/datasets/brendan45774/test-file), which contains information about passengers such as:

- Passenger class (Pclass)
- Sex
- Age
- Number of siblings/spouses aboard (SibSp)
- Number of parents/children aboard (Parch)
- Fare paid
- Port of Embarkation (Embarked)
- Survival status (target variable)

---

## 🔧 Features Used

| Feature      | Description                          |
|--------------|--------------------------------------|
| `Pclass`     | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| `Sex`        | Gender (encoded as 0 for female, 1 for male) |
| `Age`        | Age of the passenger                 |
| `SibSp`      | # of siblings/spouses aboard         |
| `Parch`      | # of parents/children aboard         |
| `Fare`       | Ticket fare                          |
| `Embarked`   | Port of Embarkation (encoded)        |

---

## 📊 Model

The project uses **Logistic Regression** as a binary classification model to predict survival (1 = Survived, 0 = Not Survived).

### 🔥 Model Performance

- Accuracy: ~100% (depending on the train/test split)

---

## 📈 Visualizations

- Confusion Matrix
- Survival count by Gender
- Correlation Heatmap

![Confusion Matrix Sample](images/confusion_matrix.png)
![Survival by Gender](images/survival_by_gender.png)

---

## 🧪 Requirements

Install the following libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
