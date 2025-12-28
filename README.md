
---

````markdown
# Smart Expense Predictor

![Website Demo:](https://isaaciyo-smart-expense-app.streamlit.app/)
This is a personal ML project that predicts monthly expenses based on user inputs.  
It was built **from scratch** using linear regression and features an **interactive Streamlit dashboard**.

---

## Features

- Linear regression model trained from scratch
- Interactive web dashboard using **Streamlit**
- Input via sliders for: Income, Rent, Utilities, and Subscriptions
- Predicts monthly expenses instantly
- **Prediction history** with download option (CSV)
- **Interactive feature weights** chart (shows influence of each input)

---

## Demo

![Dashboard Screenshots:](/screenshots/screenshot1.png)
![](/screenshots/screenshot2.png)
---

## Installation

1. Clone this repository:

```bash
git clone https://github.com/Isaaciyo/smart-expense-predictor.git
cd "Your project folder name"
````

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

Run the Streamlit dashboard:

```bash
streamlit run app.py
```

* Open your browser and interact with the input values(Rent, Income, Subscriptions etc)
* Click **Predict Expenses** to see predictions, history, and feature weights

---

## Requirements

You will need the following Python packages:

```
numpy
pandas
matplotlib
plotly
streamlit
```

> You can also generate `requirements.txt` automatically using:
>
> ```bash
> pip freeze > requirements.txt
> ```

---

## Project Structure

```
project_root/
├── data/
│   ├── data_generator.py
│   ├── expenses.csv
├── scratch/
│   ├── __init__.py
│   ├── model.py
│   ├── predict_cli.py
│   └── train.py
|   |__visualizer.py
├── sklearn_version/ (empty folder for now)
├── app.py
├── requirements.txt
├── trained_weights.npy
├── trained_bias.npy
|-- X_mean.npy
|-- X_std.npy
└── README.md
```

---

## About Me

This project was built by **Isaac Isidahome-Iyobebe**.
Explore my other projects on my GitHub: [GitHub Repository](https://github.com/Isaaciyo)

---

## License

This project is open-source and available under the MIT License.