# Expense Tracker

A smart expense tracking application built with Flask that helps you manage your finances with AI-powered insights and recommendations.

## Features

- 📊 **Expense Management**
  - Add and track daily expenses
  - Categorize expenses (Food, Transportation, Housing, Utilities, Entertainment, Other)
  - View detailed expense history
  - Smart category suggestions using Naive Bayes classification

- 📈 **Analytics & Insights**
  - Monthly expense analysis
  - Spending patterns visualization
  - Category-wise spending breakdown
  - Moving average calculations
  - Financial trend analysis

- 🤖 **AI-Powered Features**
  - Expense predictions for next month
  - Smart category suggestions based on expense descriptions
  - Personalized budget recommendations
  - Financial advice based on spending patterns

- 📱 **User Interface**
  - Clean and intuitive dashboard
  - Category icons for better visualization
  - Responsive design
  - Real-time updates

## Tech Stack

- **Backend**: Python, Flask
- **Database**: SQLite3
- **Machine Learning**: scikit-learn, NumPy
- **Frontend**: HTML, CSS, JavaScript
- **Templates**: Jinja2

## Prerequisites

- Python 3.7+
- pip (Python package manager)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/expense-tracker.git
cd expense-tracker
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

## Features in Detail

### Expense Management
- Add new expenses with date, category, amount, and description
- View all expenses in chronological order
- Automatic category suggestions based on expense descriptions

### Analytics Dashboard
- Monthly expense overview
- Category-wise spending analysis
- Spending trends and patterns
- Budget recommendations

### AI Features
- Next month expense predictions
- Smart category suggestions
- Personalized financial advice
- Spending pattern analysis

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Flask web framework
- scikit-learn for machine learning capabilities
- All contributors and users of the application 