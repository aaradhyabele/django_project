# 🛡️ Fraud Detection & Analysis System

A comprehensive, professional Django-based platform designed for real-time fraud monitoring, batch transaction analysis, and intelligent risk assessment. This system leverages Machine Learning to identify suspicious patterns and provides a robust role-based access control (RBAC) mechanism for organizational security.

---

## 🚀 Key Features

- **🧠 Multi-Model Prediction**: Utilizes Random Forest, Decision Tree, and Logistic Regression models to predict fraud.
- **📊 Interactive Dashboard**: Visualizes fraud trends and transaction patterns using Matplotlib.
- **📁 Batch Analysis**: Upload CSV datasets for bulk processing and fraud detection.
- **🕵️ Risk Scoring**: Assigns dynamic risk levels (High, Medium, Low) based on transaction amount, frequency, and time.
- **👤 Role-Based Access Control (RBAC)**: Secure access for different personas:
    - **Admin**: Full system management and user oversight.
    - **Fraud Analyst**: In-depth analysis and report generation.
    - **Compliance Officer**: Regulatory monitoring.
    - **Auditor**: Transaction verification.
    - **User**: Standard transaction monitoring.
- **📝 Historical Reporting**: Persistent storage of analysis results for auditing and trend analysis.

---

## 🛠️ Technology Stack

- **Backend**: Django 6.0
- **Database**: SQLite (Development)
- **Machine Learning**: Scikit-Learn (Logistic Regression, Decision Trees, Random Forest)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib
- **Frontend**: HTML5, Vanilla CSS, JavaScript

---

## 📥 Installation Guide

### Prerequisites
- Python 3.10+
- Git

### Steps
1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd django_project
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Database Setup**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

5. **Create Superuser (Admin)**
   ```bash
   python manage.py createsuperuser
   ```

6. **Run the Server**
   ```bash
   python manage.py runserver
   ```

---

## 📖 Usage

1. **Training**: Navigate to the home page to train the global model using a labeled dataset (CSV).
2. **Analysis**: Use the 'Fraud Analysis' feature to upload transaction logs and receive risk-scored results.
3. **Prediction**: Manually input transaction details (Amount, Frequency, Hour) to get real-time risk assessment.
4. **Reports**: View all processed records and exported analysis reports.

---

## 📁 Project Structure

```text
├── core/               # Main application logic & ML Models
│   ├── static/         # CSS & Assets
│   ├── templates/      # Dashboard and Prediction UI
│   ├── models.py       # TransactionRecord model
│   └── views.py        # ML Prediction & Analysis logic
├── user_management/    # RBAC & Profile Management
│   ├── middleware.py   # Access control logic
│   └── models.py       # User profiles and roles
├── major_pr/           # Project settings & URL configuration
├── manage.py           # Django management script
└── requirements.txt    # Python dependencies
```

---

## 🛡️ Security & Role Permissions

The system implements a custom middleware to enforce role-based restrictions. Ensure your user profile is assigned the correct role via the Admin panel to access specific analytical tools.

---

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.
