# Price Predictor Project

Welcome to the Price Predictor project! This is a web-based application designed to predict prices using machine learning techniques. The project is structured into two main directories: `Backend` and `Frontend`.

## Project Structure

```
Price_Predictor/
├── Backend/
│   ├── app.py              # Main application file
│   ├── price_model.joblib  # Trained machine learning model
│   ├── products.csv        # Dataset for training the model
│   ├── requirements.txt    # Python dependencies
│   ├── scraper.py          # Data scraping script
│   ├── streamlit_app.py    # Streamlit-based app (optional)
│   └── train_model.ipynb   # Jupyter notebook for model training
├── Frontend/
│   ├── about.html          # About page
│   ├── contact.html        # Contact page
│   ├── features.html       # Features page
│   ├── form.html           # Form page
│   ├── index.html          # Homepage
│   └── pricing.html        # Pricing page
└── README.md               # This file
```

## Overview

- **Backend**: Contains the core logic including the machine learning model, data processing scripts, and the main application file.
- **Frontend**: Includes HTML files for the user interface, providing a simple and responsive web experience.

## Getting Started

### Prerequisites
- Python 3.x
- Required Python packages (listed in `requirements.txt`)

### Installation
1. Clone the repository or download the files.
2. Navigate to the `Backend` directory.
3. Install dependencies by running:
   ```
   pip install -r requirements.txt
   ```
4. Train the model by running the `train_model.ipynb` notebook to generate `price_model.joblib`.
5. Run the application:
   ```
   python app.py
   ```

### Usage
- Open the `index.html` file in the `Frontend` directory in a web browser to access the application.
- Use the provided pages (e.g., `about.html`, `pricing.html`) for more information.

## Contributing
Feel free to fork this repository, make improvements, and submit pull requests. Contributions are welcome!

## License
This project is open-source. See the [LICENSE](LICENSE) file for more details (if applicable).
