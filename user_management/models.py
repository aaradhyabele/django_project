from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver

class Role(models.TextChoices):
    ADMIN = 'ADMIN', 'Admin'
    FRAUD_ANALYST = 'FRAUD_ANALYST', 'Fraud Analyst'
    COMPLIANCE_OFFICER = 'COMPLIANCE_OFFICER', 'Compliance Officer'
    AUDITOR = 'AUDITOR', 'Auditor'
    USER = 'USER', 'User'

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    role = models.CharField(max_length=20, choices=Role.choices, default=Role.USER)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.user.username} - {self.role}"

# Signal to create UserProfile automatically when a new User is registered
@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        role = Role.ADMIN if instance.is_superuser else Role.USER
        UserProfile.objects.get_or_create(user=instance, defaults={'role': role})
    else:
        # For existing users, ensure profile exists
        if not hasattr(instance, 'profile'):
            role = Role.ADMIN if instance.is_superuser else Role.USER
            UserProfile.objects.get_or_create(user=instance, defaults={'role': role})

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    instance.profile.save()
