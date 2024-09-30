from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import sqlite3
from datetime import datetime, timedelta
import re
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LinearRegression

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
def suggest_category(description):
    keywords = {
        'Food': ['grocery', 'restaurant', 'meal', 'food', 'eat'],
        'Transportation': ['gas', 'fuel', 'bus', 'train', 'taxi', 'uber'],
        'Housing': ['rent', 'mortgage', 'apartment'],
        'Utilities': ['electricity', 'water', 'internet', 'phone'],
        'Entertainment': ['movie', 'game', 'concert', 'show'],
    }
    
    description = description.lower()
    for category, words in keywords.items():
        if any(word in description for word in words):
            return category
    return 'Other'

# Budgeting feature
def analyze_spending(expenses):
    category_totals = defaultdict(float)
    for expense in expenses:
        category_totals[expense[2]] += expense[3]
    
    total_spent = sum(category_totals.values())
    category_percentages = {cat: (total / total_spent) * 100 for cat, total in category_totals.items()}
    
    return category_percentages

def get_monthly_expenses():
    conn = sqlite3.connect('expenses.db')
    c = conn.cursor()
    c.execute("SELECT strftime('%Y-%m', date) as month, SUM(amount) as total FROM expenses GROUP BY month ORDER BY month")
    monthly_expenses = c.fetchall()
    conn.close()
    return monthly_expenses

def predict_next_month_expense():
    monthly_expenses = get_monthly_expenses()
    if len(monthly_expenses) < 3:
        return None
    
    X = np.array(range(len(monthly_expenses))).reshape(-1, 1)
    y = np.array([expense[1] for expense in monthly_expenses])
    
    model = LinearRegression()
    model.fit(X, y)
    
    next_month = model.predict([[len(monthly_expenses)]])
    return next_month[0]

def recommend_budget():
    next_month_prediction = predict_next_month_expense()
    if next_month_prediction is None:
        return "Not enough data to make a recommendation."
    
    recommended_budget = next_month_prediction * 0.9  # Recommend 10% less than predicted
    return f"Based on your spending habits, we recommend a budget of ${recommended_budget:.2f} for next month."

@app.route('/')
def index():
    conn = sqlite3.connect('expenses.db')
    c = conn.cursor()
    c.execute("SELECT * FROM expenses ORDER BY date DESC")
    expenses = c.fetchall()
    
    formatted_expenses = []
    total = 0
    for expense in expenses:
        date = datetime.strptime(expense[1], '%Y-%m-%d').strftime('%B %d, %Y')
        formatted_expenses.append((expense[0], date, expense[2], expense[3], expense[4]))
        total += expense[3]
    
    spending_analysis = analyze_spending(expenses)
    
    conn.close()
    return render_template('index.html', expenses=formatted_expenses, total=total, spending_analysis=spending_analysis)

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
        
        # Use NLP to suggest category if not provided
        if category == 'Other':
            suggested_category = suggest_category(description)
            if suggested_category != 'Other':
                flash(f'Category suggestion: {suggested_category}', 'info')
        
        conn = sqlite3.connect('expenses.db')
        c = conn.cursor()
        c.execute("INSERT INTO expenses (date, category, amount, description) VALUES (?, ?, ?, ?)",
                  (date, category, amount, description))
        conn.commit()
        conn.close()
        
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

    return render_template('dashboard.html', 
                           spending_analysis=spending_analysis, 
                           monthly_expenses=monthly_expenses,
                           budget_recommendation=budget_recommendation,
                           recent_expenses=recent_expenses)

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

if __name__ == '__main__':
    app.run(debug=True)