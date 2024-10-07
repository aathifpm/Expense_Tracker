from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import sqlite3
from datetime import datetime, timedelta , date
import re
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LinearRegression
import math


class NaiveBayesPredictor:
    def __init__(self):
        self.category_counts = defaultdict(int)
        self.month_category_counts = defaultdict(lambda: defaultdict(int))
        self.total_months = 0

    def train(self, month, category, amount):
        self.category_counts[category] += 1
        self.month_category_counts[month][category] += 1
        self.total_months = max(self.total_months, month)

    def predict_next_month(self):
        next_month = self.total_months + 1
        predictions = {}
        total_expenses = sum(self.category_counts.values())
        
        for category in self.category_counts:
            # Calculate P(Category)
            p_category = self.category_counts[category] / total_expenses
            
            # Calculate P(NextMonth | Category)
            p_month_given_category = 1 / (self.total_months + 1)  # Assume uniform distribution
            
            # Combine probabilities
            predictions[category] = p_category * p_month_given_category
        
        # Normalize probabilities
        total_prob = sum(predictions.values())
        for category in predictions:
            predictions[category] /= total_prob
        
        return predictions

# Initialize the predictor
predictor = NaiveBayesPredictor()

# Train the predictor with existing data
def train_predictor():
    conn = sqlite3.connect('expenses.db')
    c = conn.cursor()
    c.execute("SELECT strftime('%Y-%m', date) as month, category, amount FROM expenses")
    for month, category, amount in c.fetchall():
        month_num = int(month.split('-')[1])  # Extract month number
        predictor.train(month_num, category, amount)
    conn.close()

def predict_next_month_expense():
    predictions = predictor.predict_next_month()
    
    # Convert predictions to a more readable format
    result = []
    for category, probability in predictions.items():
        result.append(f"{category}: {probability:.2%}")
    
    return "\n".join(result)

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Add this line for flash messages

# Database initialization
def init_db():
    conn = sqlite3.connect('expenses.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS expenses
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  date TEXT,
                  category TEXT,
                  amount REAL,
                  description TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Simple NLP for category suggestion
class NaiveBayesClassifier:
    def __init__(self):
        self.categories = set()
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.category_counts = defaultdict(int)
        self.total_count = 0

    def train(self, description, category):
        self.categories.add(category)
        self.category_counts[category] += 1
        self.total_count += 1
        
        words = self._tokenize(description)
        for word in words:
            self.word_counts[category][word] += 1

    def predict(self, description):
        words = self._tokenize(description)
        scores = {}
        
        for category in self.categories:
            score = math.log(self.category_counts[category] / self.total_count)
            
            for word in words:
                word_count = self.word_counts[category][word]
                word_prob = (word_count + 1) / (sum(self.word_counts[category].values()) + len(self.word_counts[category]))
                score += math.log(word_prob)
            
            scores[category] = score
        
        return max(scores, key=scores.get)

    def _tokenize(self, text):
        text = text.lower()
        return re.findall(r'\w+', text)

# Initialize the classifier
classifier = NaiveBayesClassifier()

# Train the classifier with existing data
def train_classifier():
    conn = sqlite3.connect('expenses.db')
    c = conn.cursor()
    c.execute("SELECT description, category FROM expenses")
    for description, category in c.fetchall():
        classifier.train(description, category)
    conn.close()

# Call this function when the app starts
train_classifier()

def suggest_category(description):
    return classifier.predict(description)

# Budgeting feature
def analyze_spending(expenses):
    category_totals = defaultdict(float)
    for expense in expenses:
        category_totals[expense[2]] += expense[3]
    
    total_spent = sum(category_totals.values())
    category_percentages = {cat: (total / total_spent) * 100 for cat, total in category_totals.items()}
    
    return category_percentages

def get_monthly_expenses(months=12):
    conn = sqlite3.connect('expenses.db')
    c = conn.cursor()
    c.execute("""
        SELECT 
            strftime('%Y-%m', date) as month, 
            SUM(amount) as total,
            AVG(SUM(amount)) OVER (ORDER BY strftime('%Y-%m', date) ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) as moving_avg
        FROM expenses
        WHERE date >= date('now', '-' || ? || ' months')
        GROUP BY month
        ORDER BY month
    """, (months,))
    monthly_expenses = c.fetchall()
    conn.close()
    return monthly_expenses

def recommend_budget():
    next_month_prediction = predict_next_month_expense()
    if next_month_prediction is None:
        return "Not enough data to make a recommendation."
    
    # Convert the prediction string to a float
    predicted_amount = float(next_month_prediction.split(":")[1].strip("%").replace(",", ""))
    
    recommended_budget = predicted_amount * 0.9  # Recommend 10% less than predicted
    return f"Based on your spending habits, we recommend a budget of ${recommended_budget:.2f} for next month."


@app.route('/')
def index():
    conn = sqlite3.connect('expenses.db')
    c = conn.cursor()
    c.execute("SELECT id, date, category, amount, description FROM expenses ORDER BY date DESC")
    expenses = c.fetchall()
    
    # Convert expenses to a list of dictionaries for easier JSON serialization
    expenses_list = [
        {
            'id': e[0],
            'date': e[1],
            'category': e[2],
            'amount': e[3],
            'description': e[4]
        } for e in expenses
    ]
    
    spending_analysis = analyze_spending(expenses)
    
    # Get unique categories
    c.execute("SELECT DISTINCT category FROM expenses")
    categories = [row[0] for row in c.fetchall()]
    
    # Calculate monthly summary
    today = date.today()
    first_day_of_month = date(today.year, today.month, 1)
    c.execute("""
        SELECT SUM(amount) as total, COUNT(*) as count, MAX(category) as top_category
        FROM expenses
        WHERE date >= ?
    """, (first_day_of_month.strftime('%Y-%m-%d'),))
    monthly_summary = c.fetchone()
    
    this_month_total = monthly_summary[0] if monthly_summary[0] else 0
    avg_daily_spending = this_month_total / today.day if this_month_total > 0 else 0
    top_category = monthly_summary[2] if monthly_summary[2] else 'N/A'
    
    conn.close()
    return render_template('index.html', 
                           expenses=expenses_list, 
                           total=sum(e['amount'] for e in expenses_list),
                           spending_analysis=spending_analysis, 
                           categories=categories,
                           this_month_total=this_month_total,
                           avg_daily_spending=avg_daily_spending,
                           top_category=top_category)

@app.route('/add', methods=['GET', 'POST'])
def add_expense():
    categories = ['Food', 'Transportation', 'Housing', 'Utilities', 'Entertainment', 'Other']
    
    if request.method == 'POST':
        date = request.form['date']
        category = request.form['category']
        amount = request.form['amount']
        description = request.form['description']
        
        # Data validation
        if not date or not category or not amount or not description:
            flash('All fields are required', 'error')
            return render_template('add.html', categories=categories)
        
        try:
            amount = float(amount)
            if amount <= 0:
                raise ValueError
        except ValueError:
            flash('Amount must be a positive number', 'error')
            return render_template('add.html', categories=categories)
        
        # Use Naive Bayes to suggest category
        suggested_category = suggest_category(description)
        if category == 'Other' and suggested_category in categories:
            flash(f'Category suggestion: {suggested_category}', 'info')
        
        conn = sqlite3.connect('expenses.db')
        c = conn.cursor()
        c.execute("INSERT INTO expenses (date, category, amount, description) VALUES (?, ?, ?, ?)",
                  (date, category, amount, description))
        conn.commit()
        conn.close()
        
        # Train the classifier with the new data
        classifier.train(description, category)
        
        flash('Expense added successfully', 'success')
        return redirect(url_for('index'))
    
    return render_template('add.html', categories=categories)

# Add this function to your app.py
def get_category_icon(category):
    icons = {
        'Food': 'utensils',
        'Transportation': 'car',
        'Housing': 'home',
        'Utilities': 'bolt',
        'Entertainment': 'film',
        'Other': 'question'
    }
    return icons.get(category, 'tag')

# Add this line after creating the Flask app
app.jinja_env.globals.update(get_category_icon=get_category_icon)

@app.route('/dashboard')
def dashboard():
    conn = sqlite3.connect('expenses.db')
    c = conn.cursor()
    c.execute("SELECT * FROM expenses ORDER BY date DESC")
    expenses = c.fetchall()
    conn.close()

    spending_analysis = analyze_spending(expenses)
    monthly_expenses = get_monthly_expenses()
    budget_recommendation = calculate_budget_recommendation()
    recent_expenses = get_recent_expenses()
    
    # Use the NaiveBayesPredictor
    train_predictor()
    next_month_prediction = predict_next_month_expense()
    
    # Get financial advice
    financial_advice = get_financial_advice()

    return render_template('dashboard.html', 
                           spending_analysis=spending_analysis, 
                           monthly_expenses=monthly_expenses,
                           budget_recommendation=budget_recommendation,
                           recent_expenses=recent_expenses,
                           next_month_prediction=next_month_prediction,
                           financial_advice=financial_advice)

@app.route('/api/expenses')
def api_expenses():
    conn = sqlite3.connect('expenses.db')
    c = conn.cursor()
    c.execute("SELECT * FROM expenses ORDER BY date DESC")
    expenses = c.fetchall()
    conn.close()

    return jsonify([{
        'id': e[0],
        'date': e[1],
        'category': e[2],
        'amount': e[3],
        'description': e[4]
    } for e in expenses])

def calculate_budget_recommendation():
    conn = sqlite3.connect('expenses.db')
    c = conn.cursor()
    c.execute("SELECT category, SUM(amount) FROM expenses GROUP BY category")
    category_totals = c.fetchall()
    conn.close()

    total_expenses = sum(amount for _, amount in category_totals)
    avg_monthly_expenses = total_expenses / 3  # Assuming 3 months of data

    # Simple recommendation: 10% less than average monthly expenses
    recommended_budget = avg_monthly_expenses * 0.9

    return f"Based on your spending habits, we recommend a monthly budget of ${recommended_budget:.2f}"

def get_recent_expenses(limit=5):
    conn = sqlite3.connect('expenses.db')
    c = conn.cursor()
    c.execute("SELECT * FROM expenses ORDER BY date DESC LIMIT ?", (limit,))
    recent_expenses = c.fetchall()
    conn.close()
    return [{'date': e[1], 'category': e[2], 'amount': e[3], 'description': e[4]} for e in recent_expenses]

class ExpenseExpertSystem:
    def __init__(self):
        self.rules = [
            (lambda data: data['total_expenses'] > data['income'], 
             "Your expenses exceed your income. Consider reducing non-essential spending."),
            (lambda data: data['savings_rate'] < 0.1, 
             "Your savings rate is low. Aim to save at least 10% of your income."),
            (lambda data: data['largest_category'][1] > 0.5 * data['total_expenses'], 
             lambda data: f"Your {data['largest_category'][0]} expenses are over 50% of your total. Look for ways to reduce this category."),
            (lambda data: data['expense_trend'] > 1.1, 
             "Your expenses are trending upwards. Review your spending habits."),
            (lambda data: data['expense_trend'] < 0.9, 
             "Great job! Your expenses are trending downwards."),
        ]

    def get_advice(self, user_data):
        advice = []
        for condition, recommendation in self.rules:
            if condition(user_data):
                if callable(recommendation):
                    advice.append(recommendation(user_data))
                else:
                    advice.append(recommendation)
        return advice

def analyze_expenses():
    conn = sqlite3.connect('expenses.db')
    c = conn.cursor()
    
    # Get total expenses
    c.execute("SELECT SUM(amount) FROM expenses")
    total_expenses = c.fetchone()[0] or 0

    # Get largest expense category
    c.execute("SELECT category, SUM(amount) FROM expenses GROUP BY category ORDER BY SUM(amount) DESC LIMIT 1")
    largest_category = c.fetchone() or ('Unknown', 0)

    # Get expense trend (compare last month to previous month)
    c.execute("""
        SELECT 
            SUM(CASE WHEN date >= date('now', '-1 month') THEN amount ELSE 0 END) as last_month,
            SUM(CASE WHEN date >= date('now', '-2 month') AND date < date('now', '-1 month') THEN amount ELSE 0 END) as previous_month
        FROM expenses
    """)
    last_month, previous_month = c.fetchone()
    expense_trend = (last_month or 0) / (previous_month or 1)  # Avoid division by zero

    conn.close()

    return {
        'total_expenses': total_expenses,
        'income': 5000,  # This should be provided by the user or stored in the database
        'savings_rate': (5000 - total_expenses) / 5000,  # Assuming income is 5000
        'largest_category': largest_category,
        'expense_trend': expense_trend
    }

expert_system = ExpenseExpertSystem()

def get_financial_advice():
    user_data = analyze_expenses()
    advice = expert_system.get_advice(user_data)
    return "\n".join(advice) if advice else "Your finances look good! Keep it up!"

# You can call this function from your route handlers
# result = get_financial_advice()

@app.route('/api/monthly_expenses/<int:months>')
def api_monthly_expenses(months):
    monthly_expenses = get_monthly_expenses(months)
    return jsonify(monthly_expenses)

@app.route('/api/suggest_category', methods=['POST'])
def api_suggest_category():
    description = request.json.get('description', '')
    suggested_category = suggest_category(description)
    return jsonify({'suggested_category': suggested_category})

if __name__ == '__main__':
    app.run(debug=True)