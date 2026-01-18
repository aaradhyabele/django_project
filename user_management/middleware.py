from django.shortcuts import redirect
from django.urls import reverse
from django.contrib import messages
from .models import Role

class RoleBasedAccessMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if not request.user.is_authenticated:
            return self.get_response(request)

        # Get user role (safe access)
        if request.user.is_superuser:
            role = Role.ADMIN
        else:
            try:
                role = request.user.profile.role
            except Exception:
                role = Role.USER

        path = request.path

        # RBAC Rules Enforcement
        # Admin → full system access (no restrictions)
        if role == Role.ADMIN:
            return self.get_response(request)

        # Prediction & Analysis Access
        if path.startswith(reverse('fraud_analysis')):
            if role != Role.FRAUD_ANALYST:
                messages.error(request, "Access denied: Fraud Analyst role required for bulk analysis.")
                return redirect('home')

        if path.startswith(reverse('predict')):
            # Allow USER role to access quick prediction as requested
            if role not in [Role.FRAUD_ANALYST, Role.USER]:
                messages.error(request, "Access denied: Quick Prediction requires Fraud Analyst or User role.")
                return redirect('home')

        # Compliance Officer / Auditor → reports & transaction history
        if path.startswith(reverse('report')):
            if role not in [Role.COMPLIANCE_OFFICER, Role.AUDITOR]:
                messages.error(request, "Access denied: Compliance/Auditor role required.")
                return redirect('home')

        # User Management → Allow the app's own decorators to handle it
        if path.startswith('/user-management/'):
             return self.get_response(request)

        return self.get_response(request)
