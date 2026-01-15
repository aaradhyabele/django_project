from django.shortcuts import render
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm


from .models import TransactionRecord

@login_required
def home(request):
    return render(request, 'core/home.html')

# Register view
def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)  # auto-login after registration
            messages.success(request, "Registration successful!")
            return redirect('home')
        else:
            messages.error(request, "Registration failed. Please correct the errors.")
    else:
        form = UserCreationForm()
    return render(request, 'core/register.html', {'form': form})

# Login view
def user_login(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f"Welcome {username}!")
                return redirect('home')
            else:
                messages.error(request, "Invalid username or password")
        else:
            messages.error(request, "Invalid username or password")
    else:
        form = AuthenticationForm()
    return render(request, 'core/login.html', {'form': form})

# Logout view
@login_required
def user_logout(request):
    logout(request)
    messages.info(request, "Logged out successfully!")
    return redirect('login')

# ================= GLOBAL VARIABLES =================
lr = None
dt = None
rf = None  # Random Forest
scaler = None
X_test_scaled = None
y_test = None
model_trained = False


def preprocess_data(df):
    """
    Applies feature engineering:
    1. amount_per_txn
    2. is_night
    3. high_amount_flag
    """
    df = df.copy()
    df['amount_per_txn'] = df['amount'] / (df['transactions'] + 1)
    df['is_night'] = df['hour'].apply(lambda x: 1 if x < 6 else 0)
    df['high_amount_flag'] = (df['amount'] > 50000).astype(int)
    return df


def about(request):
    return render(request, 'core/about.html')


def report(request):
    # Fetch all records, newest first
    report_data = TransactionRecord.objects.all().order_by('-timestamp')
    return render(request, 'core/report.html', {'report_data': report_data})


def fraud_prediction(request):
    global lr, dt, rf, scaler, X_test_scaled, y_test, model_trained

    context = {
        "result": None,
        "error": None,
        "message": None
    }

    if request.method == "POST":

        # ========== TRAIN MODEL ==========
        if "train_csv" in request.FILES:
            try:
                df = pd.read_csv(request.FILES["train_csv"])

                # Feature Engineering
                df = preprocess_data(df)

                # Select Features (Update to include new ones)
                features = ["amount", "transactions", "hour", "amount_per_txn", "is_night", "high_amount_flag"]
                X = df[features]
                y = df["fraud"]

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )

                # Logistic Regression
                lr = LogisticRegression()
                lr.fit(X_train, y_train)

                # Decision Tree
                dt = DecisionTreeClassifier()
                dt.fit(X_train, y_train)

                # Random Forest
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X_train, y_train)

                X_test_scaled = X_test
                model_trained = True

                context["message"] = "Models trained successfully (LR, DT, RF)"

            except Exception as e:
                context["error"] = str(e)

        # ========== PREDICT TRANSACTION ==========
        elif model_trained:
            try:
                amount = float(request.POST.get("amount"))
                transactions = int(request.POST.get("transactions"))
                hour = int(request.POST.get("hour"))

                # Create DataFrame for single user input
                user_df = pd.DataFrame([{
                    "amount": amount,
                    "transactions": transactions,
                    "hour": hour
                }])

                # Apply same Feature Engineering
                user_df = preprocess_data(user_df)
                
                features = ["amount", "transactions", "hour", "amount_per_txn", "is_night", "high_amount_flag"]
                user_scaled = scaler.transform(user_df[features])

                pred_lr = lr.predict(user_scaled)[0]
                pred_dt = dt.predict(user_scaled)[0]
                pred_rf = rf.predict(user_scaled)[0]

                acc_lr = round(accuracy_score(y_test, lr.predict(X_test_scaled)), 2)
                acc_dt = round(accuracy_score(y_test, dt.predict(X_test_scaled)), 2)
                acc_rf = round(accuracy_score(y_test, rf.predict(X_test_scaled)), 2)

                # ===== Risk Score calculation (Simplified/Heuristic) =====
                risk_score = 0
                if amount > 1000: risk_score += 0.3
                if transactions > 2: risk_score += 0.2
                if hour >= 22 or hour <= 5: risk_score += 0.3
                
                # Boost risk if RF predicts fraud (it's usually robust)
                if pred_rf == 1: risk_score += 0.2
                
                risk_score = min(risk_score, 1.0)

                if risk_score >= 0.7:
                    risk_level = "High Risk"
                    final_decision = "Fraudulent"
                    fraud_result = "Fraud"
                elif risk_score >= 0.4:
                    risk_level = "Medium Risk"
                    final_decision = "Suspicious"
                    fraud_result = "Normal" # Or Suspicious
                else:
                    risk_level = "Low Risk"
                    final_decision = "Normal"
                    fraud_result = "Normal"

                # === SAVE TO DB ===
                TransactionRecord.objects.create(
                    amount=amount,
                    transactions=transactions,
                    hour=hour,
                    fraud_result=fraud_result, # Simple status
                    risk_level=risk_level,
                    risk_score=risk_score
                )

                context["result"] = {
                    "prediction_logistic": "Fraud" if pred_lr == 1 else "Normal",
                    "accuracy_logistic": acc_lr,
                    "prediction_dt": "Fraud" if pred_dt == 1 else "Normal",
                    "accuracy_dt": acc_dt,
                    "prediction_rf": "Fraud" if pred_rf == 1 else "Normal",
                    "accuracy_rf": acc_rf,
                    "risk_level": risk_level,
                    "final_decision": final_decision,
                    "risk_score": risk_score
                }

            except Exception as e:
                context["error"] = str(e)

        else:
            context["error"] = "Please train the model first."

    return render(request, "core/predict.html", context)

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from django.shortcuts import render
from django.conf import settings

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ===== GLOBAL VARIABLES (Model trained only once) =====
model = None
scaler = None
model_trained = False


def fraud_analysis(request):
    global model, scaler, model_trained

    context = {}

    if request.method == "POST":

        # ========= STEP 1: TRAIN MODEL =========
        if "train_csv" in request.FILES:
            train_file = request.FILES["train_csv"]

            df = pd.read_csv(train_file)

            X = df[["amount", "transactions", "hour"]]
            y = df["fraud"]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = LogisticRegression()
            model.fit(X_scaled, y)

            model_trained = True
            context["message"] = "Model trained successfully"

        #

    # ========= STEP 2: ANALYZE USER CSV =========
    if request.method == "POST" and "user_csv" in request.FILES and model_trained:

        # ===== READ USER CSV =====
        user_file = request.FILES["user_csv"]
        user_df = pd.read_csv(user_file)

        # ===== DATA CLEANING =====
        # Ensure numeric columns
        for col in ["amount", "transactions", "hour"]:
            user_df[col] = pd.to_numeric(user_df[col], errors='coerce')

        # Drop rows with missing critical values
        user_df.dropna(subset=["amount", "transactions", "hour"], inplace=True)

        # Filter out invalid values
        user_df = user_df[
            (user_df["amount"] >= 0) &
            (user_df["transactions"] > 0) &
            (user_df["hour"].between(0, 23))
        ]

        # ===== FEATURE SELECTION & SCALING =====
        X_user = user_df[["amount", "transactions", "hour"]]
        X_user_scaled = scaler.transform(X_user)

        # ===== PREDICTION =====
        predictions = model.predict(X_user_scaled)
        probabilities = model.predict_proba(X_user_scaled)

        # ===== ADD RESULTS TO DATAFRAME =====
        user_df["predicted_fraud"] = predictions
        user_df["fraud_prob"] = probabilities[:, 1]


        # ===== DATASET SUMMARY =====
        total_transactions = len(user_df)
        fraud_count = int((user_df["predicted_fraud"] == 1).sum())
        normal_count = total_transactions - fraud_count
        fraud_rate = round((fraud_count / total_transactions) * 100, 2)

        context["dataset_summary"] = {
            "total": total_transactions,
            "fraud": fraud_count,
            "normal": normal_count,
            "fraud_rate": fraud_rate
        }

        # ===== Fraud Count vs Time =====
        fraud_time = (
            user_df[user_df["predicted_fraud"] == 1]
            .groupby("hour")
            .size()
        )

        plt.figure()
        fraud_time.plot()
        plt.xlabel("Hour")
        plt.ylabel("Fraud Count")
        plt.title("Fraud Count vs Time")

        fraud_time_path = os.path.join(settings.MEDIA_ROOT, "fraud_time.png")
        plt.savefig(fraud_time_path)
        plt.close()

        # ===== Fraud vs Normal Probability =====
        fraud_prob = user_df["fraud_prob"].mean()
        normal_prob = 1 - fraud_prob

        plt.figure()
        plt.bar(["Fraud", "Normal"], [fraud_prob, normal_prob])
        plt.ylabel("Probability")
        plt.title("Fraud vs Normal Probability")

        prob_path = os.path.join(settings.MEDIA_ROOT, "fraud_prob.png")
        plt.savefig(prob_path)
        plt.close()

        context["fraud_time_chart"] = "fraud_time.png"
        context["prob_chart"] = "fraud_prob.png"
        context["analysis_done"] = True

        # ===== FRAUD RECORDS & EXPORT =====
        # Filter only fraud records
        fraud_df = user_df[user_df["predicted_fraud"] == 1].copy()
        
        # Convert to list of dicts for the template
        fraud_records = fraud_df.to_dict(orient="records")
        
        # Round probability and Calculate Risk Level
        for record in fraud_records:
            # Rounding
            record['fraud_prob'] = round(record['fraud_prob'], 4)
            
            # Risk Calc
            amount = record.get('amount', 0)
            transactions = record.get('transactions', 0)
            hour = record.get('hour', 0)
            
            risk_score = 0
            if amount > 1000: risk_score += 0.3
            if transactions > 2: risk_score += 0.2
            if hour >= 22 or hour <= 5: risk_score += 0.3
            risk_score = min(risk_score, 1.0)

            if risk_score >= 0.7:
                record['risk_level'] = "High"
            elif risk_score >= 0.4:
                record['risk_level'] = "Medium"
            else:
                record['risk_level'] = "Low"

        context["fraud_records"] = fraud_records

        # Store in session for CSV export
        request.session['fraud_data'] = fraud_records

    return render(request, "core/fraud_analysis.html", context)


from django.http import HttpResponse
import csv

def export_fraud_csv(request):
    """
    Exports the fraud data stored in session to a CSV file.
    """
    fraud_data = request.session.get('fraud_data', [])

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="fraud_detected.csv"'

    writer = csv.writer(response)
    # Write header
    writer.writerow(['Amount', 'Transactions', 'Hour', 'Fraud Probability', 'Risk Level'])

    # Write data
    for row in fraud_data:
        writer.writerow([
            row.get('amount'), 
            row.get('transactions'), 
            row.get('hour'), 
            row.get('fraud_prob'),
            row.get('risk_level')
        ])

    return response


