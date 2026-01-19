from django.shortcuts import render, redirect
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from django.contrib import messages
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from django.conf import settings
from .models import TransactionRecord
from django.http import HttpResponse
import csv
import pickle

# ================= GLOBAL MODEL VARIABLES =================
MODEL_PATH = os.path.join(settings.BASE_DIR, 'core', 'global_model.pkl')

g_lr = None
g_dt = None
g_rf = None
g_scaler = None
g_model_trained = False
g_feature_cols = ["amount", "transactions", "hour"]

# Accuracy scores
g_accuracy_lr = None
g_accuracy_dt = None
g_accuracy_rf = None

def load_global_model():
    """Loads the model from disk if it exists."""
    global g_lr, g_dt, g_rf, g_scaler, g_model_trained, g_accuracy_lr, g_accuracy_dt, g_accuracy_rf
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                data = pickle.load(f)
                g_lr = data.get('lr')
                g_dt = data.get('dt')
                g_rf = data.get('rf')
                g_scaler = data.get('scaler')
                g_accuracy_lr = data.get('accuracy_lr', 0)
                g_accuracy_dt = data.get('accuracy_dt', 0)
                g_accuracy_rf = data.get('accuracy_rf', 0)
                g_model_trained = True
                print("Global model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")

# Try loading on startup
load_global_model()


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

@login_required
def user_logout(request):
    logout(request)
    messages.info(request, "Logged out successfully!")
    return redirect('login')


# ================= GLOBAL TRAINING VIEW =================
@login_required
def train_global_model(request):
    global g_lr, g_dt, g_rf, g_scaler, g_model_trained, g_accuracy_lr, g_accuracy_dt, g_accuracy_rf
    
    if request.method == 'POST' and 'train_csv' in request.FILES:
        try:
            csv_file = request.FILES['train_csv']
            df = pd.read_csv(csv_file)
            
            # Simple Preprocessing
            target = 'fraud'
            
            # Check basic columns first
            required = g_feature_cols + [target]
            if not all(col in df.columns for col in required):
                messages.error(request, f"CSV must contain columns: {required}")
                return redirect(request.META.get('HTTP_REFERER', 'home'))

            df = df.dropna(subset=required)
            
            X = df[g_feature_cols]
            y = df[target]
            
            # Split data for training and testing
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            g_scaler = StandardScaler()
            X_train_scaled = g_scaler.fit_transform(X_train)
            X_test_scaled = g_scaler.transform(X_test)
            
            # Train Logistic Regression
            g_lr = LogisticRegression()
            g_lr.fit(X_train_scaled, y_train)
            g_accuracy_lr = round(accuracy_score(y_test, g_lr.predict(X_test_scaled)) * 100, 2)
            
            # Train Decision Tree
            g_dt = DecisionTreeClassifier()
            g_dt.fit(X_train_scaled, y_train)
            g_accuracy_dt = round(accuracy_score(y_test, g_dt.predict(X_test_scaled)) * 100, 2)
            
            # Train Random Forest
            g_rf = RandomForestClassifier(n_estimators=100, random_state=42)
            g_rf.fit(X_train_scaled, y_train)
            g_accuracy_rf = round(accuracy_score(y_test, g_rf.predict(X_test_scaled)) * 100, 2)
            
            g_model_trained = True
            
            # Save to disk (including accuracy scores)
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump({
                    'lr': g_lr,
                    'dt': g_dt,
                    'rf': g_rf,
                    'scaler': g_scaler,
                    'accuracy_lr': g_accuracy_lr,
                    'accuracy_dt': g_accuracy_dt,
                    'accuracy_rf': g_accuracy_rf
                }, f)
                
            messages.success(request, "Global Model Trained Successfully!")
            
        except Exception as e:
            messages.error(request, f"Training failed: {str(e)}")
            
    return redirect(request.META.get('HTTP_REFERER', 'home'))


def about(request):
    return render(request, 'core/about.html')


def report(request):
    # Fetch all records, newest first
    report_data = TransactionRecord.objects.all().order_by('-timestamp')
    return render(request, 'core/report.html', {'report_data': report_data})


@login_required
def fraud_prediction(request):
    global g_lr, g_dt, g_rf, g_scaler, g_model_trained

    context = {
        "result": None,
        "error": None,
        "message": None,
        "model_trained": g_model_trained
    }

    if request.method == "POST":
        if not g_model_trained:
             context["error"] = "Please train the model first using the box at the bottom."
        
        else:
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

                # Scale
                user_scaled = g_scaler.transform(user_df[g_feature_cols])

                pred_lr = g_lr.predict(user_scaled)[0]
                pred_dt = g_dt.predict(user_scaled)[0]
                pred_rf = g_rf.predict(user_scaled)[0]

                # Risk Score calculation (Simplified/Heuristic)
                risk_score = 0
                if amount > 10000: risk_score += 0.4
                if transactions > 5: risk_score += 0.3
                if hour >= 22 or hour <= 5: risk_score += 0.3
                
                if pred_rf == 1: risk_score += 0.2
                
                risk_score = min(risk_score, 1.0)

                if risk_score >= 0.7:
                    risk_level = "High Risk"
                    final_decision = "Fraudulent"
                    fraud_result = "Fraud"
                elif risk_score >= 0.4:
                    risk_level = "Medium Risk"
                    final_decision = "Suspicious"
                    fraud_result = "Normal" 
                else:
                    risk_level = "Low Risk"
                    final_decision = "Normal"
                    fraud_result = "Normal"

                # Reason Logic
                reasons = []
                if amount > 10000: reasons.append("High Amount")
                if transactions > 5: reasons.append("High Frequency")
                if hour >= 22 or hour <= 5: reasons.append("Odd Hour")
                
                # Enhanced Reason Logic
                if not reasons:
                    if pred_rf == 1 and risk_level != "Low Risk":
                        fraud_reason = "Model Flagged Suspicious"
                    else:
                        fraud_reason = "None"
                else:
                    fraud_reason = " & ".join(reasons)

                # === SAVE TO DB ===
                TransactionRecord.objects.create(
                    amount=amount,
                    transactions=transactions,
                    hour=hour,
                    fraud_result=fraud_result,
                    risk_level=risk_level,
                    risk_score=risk_score,
                    fraud_reason=fraud_reason
                )

                context["result"] = {
                    "prediction_logistic": "Fraud" if pred_lr == 1 else "Normal",
                    "prediction_dt": "Fraud" if pred_dt == 1 else "Normal",
                    "prediction_rf": "Fraud" if pred_rf == 1 else "Normal",
                    "accuracy_logistic": f"{g_accuracy_lr}%" if g_accuracy_lr else "N/A",
                    "accuracy_dt": f"{g_accuracy_dt}%" if g_accuracy_dt else "N/A",
                    "accuracy_rf": f"{g_accuracy_rf}%" if g_accuracy_rf else "N/A",
                    "risk_level": risk_level,
                    "final_decision": final_decision,
                    "risk_score": risk_score,
                    "fraud_reason": fraud_reason
                }

            except Exception as e:
                context["error"] = str(e)

    return render(request, "core/predict.html", context)


@login_required
def fraud_analysis(request):
    global g_model_trained, g_scaler, g_lr

    context = {"model_trained": g_model_trained}

    if request.method == "POST" and "user_csv" in request.FILES:
        if not g_model_trained:
             context["error"] = "Please train the model first using the box at the bottom."
             return render(request, "core/fraud_analysis.html", context)

        # ===== READ USER CSV =====
        try:
            user_file = request.FILES["user_csv"]
            user_df = pd.read_csv(user_file)

            # ===== DATA CLEANING =====
            for col in g_feature_cols:
                if col not in user_df.columns:
                     context["error"] = f"Missing column: {col}"
                     return render(request, "core/fraud_analysis.html", context)
                user_df[col] = pd.to_numeric(user_df[col], errors='coerce')

            user_df.dropna(subset=g_feature_cols, inplace=True)
            
            # Simple Filter
            user_df = user_df[
                (user_df["amount"] >= 0) &
                (user_df["transactions"] >= 0)
            ]

            # ===== SCALING =====
            X_user = user_df[g_feature_cols]
            X_user_scaled = g_scaler.transform(X_user)

            # ===== PREDICTION =====
            model = g_lr # Defaulting to LR for analysis
            predictions = model.predict(X_user_scaled)
            probabilities = model.predict_proba(X_user_scaled)

            # ===== ADD RESULTS =====
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

            # ===== CHARTS =====
            if "hour" in user_df.columns:
                fraud_time = (
                    user_df[user_df["predicted_fraud"] == 1]
                    .groupby("hour")
                    .size()
                )
                plt.figure()
                fraud_time.plot(kind='bar')
                plt.xlabel("Hour")
                plt.ylabel("Fraud Count")
                plt.title("Fraud Count vs Time")
                os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
                fraud_time_path = os.path.join(settings.MEDIA_ROOT, "fraud_time.png")
                plt.savefig(fraud_time_path)
                plt.close()
                context["fraud_time_chart"] = "fraud_time.png"

            fraud_prob = user_df["fraud_prob"].mean()
            normal_prob = 1 - fraud_prob
            
            plt.figure()
            plt.bar(["Fraud", "Normal"], [fraud_prob, normal_prob])
            plt.ylabel("Probability")
            plt.title("Fraud vs Normal Probability")
            os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
            prob_path = os.path.join(settings.MEDIA_ROOT, "fraud_prob.png")
            plt.savefig(prob_path)
            plt.close()

            context["prob_chart"] = "fraud_prob.png"
            context["analysis_done"] = True

            # ===== EXPORT DATA =====
            fraud_df = user_df[user_df["predicted_fraud"] == 1].copy()
            fraud_records = fraud_df.to_dict(orient="records")
            
            for record in fraud_records:
                record['fraud_prob'] = round(record.get('fraud_prob', 0), 4)
                
                # Risk Logic
                risk_score = record.get('fraud_prob', 0) # Use prob as base risk
                record['risk_level'] = "High" if risk_score > 0.7 else "Medium" if risk_score > 0.4 else "Low"

                # Reason Logic
                reasons = []
                amount = record.get('amount', 0)
                transactions = record.get('transactions', 0)
                hour = record.get('hour', 0)

                if amount > 10000:
                    reasons.append("High Amount")
                if transactions > 5:
                    reasons.append("High Frequency")
                if hour >= 22 or hour <= 5:
                    reasons.append("Odd Hour")

                if not reasons:
                    record['fraud_reason'] = "Multiple suspicious patterns"
                elif len(reasons) == 1:
                    record['fraud_reason'] = reasons[0]
                else:
                    record['fraud_reason'] = f"{reasons[0]} & {reasons[1]}"

            context["fraud_records"] = fraud_records
            request.session['fraud_data'] = fraud_records

        except Exception as e:
            context["error"] = f"Error processing file: {str(e)}"

    return render(request, "core/fraud_analysis.html", context)


def export_fraud_csv(request):
    """
    Exports the fraud data stored in session to a CSV file.
    """
    fraud_data = request.session.get('fraud_data', [])

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="fraud_detected.csv"'

    writer = csv.writer(response)
    writer.writerow(['Amount', 'Transactions', 'Hour', 'Fraud Probability', 'Risk Level', 'Reason'])

    for row in fraud_data:
        writer.writerow([
            row.get('amount'), 
            row.get('transactions'), 
            row.get('hour'), 
            row.get('fraud_prob'),
            row.get('risk_level'),
            row.get('fraud_reason')
        ])

    return response
