from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from .models import UserProfile, Role

def admin_only(view_func):
    """Simple decorator for admin access."""
    def _wrapped_view(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect('login')
        if request.user.is_superuser:
            return view_func(request, *args, **kwargs)
            
        try:
            if request.user.profile.role != Role.ADMIN:
                messages.error(request, "Access denied: Admin role required.")
                return redirect('home')
        except Exception:
            return redirect('login')
        return view_func(request, *args, **kwargs)
    return _wrapped_view

@login_required
@admin_only
def user_list(request):
    users = User.objects.all().select_related('profile')
    return render(request, 'user_management/user_list.html', {'users': users})

@login_required
@admin_only
def assign_role(request, user_id):
    managed_user = get_object_or_404(User, id=user_id)
    if request.method == 'POST':
        new_role = request.POST.get('role')
        is_active = request.POST.get('is_active') == 'on'
        
        profile = managed_user.profile
        profile.role = new_role
        profile.is_active = is_active
        profile.save()
        
        # Also update the Django User's is_active status
        managed_user.is_active = is_active
        managed_user.save()
        
        messages.success(request, f"Updated permissions for {managed_user.username}")
        return redirect('user_list')
    
    return render(request, 'user_management/assign_role.html', {
        'managed_user': managed_user,
        'roles': Role.choices
    })

@login_required
@admin_only
def user_create(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            # Default profile is created by signal
            messages.success(request, f"User {user.username} created successfully. You can now assign a role.")
            return redirect('user_list')
    else:
        form = UserCreationForm()
    return render(request, 'user_management/user_create.html', {'form': form})
