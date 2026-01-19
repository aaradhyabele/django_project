from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse
from .models import UserProfile, Role

class MiddlewareAccessTest(TestCase):
    def setUp(self):
        # Create a non-admin user
        self.user = User.objects.create_user(username='testuser', password='password123')
        self.user.profile.role = Role.USER
        self.user.profile.save()

        # Create an admin user
        self.admin_user = User.objects.create_superuser(username='adminuser', password='password123', email='admin@test.com')

        self.client = Client()

    def test_non_admin_user_blocked_from_user_list(self):
        """
        Verify that a non-admin user is redirected from the user management area.
        """
        self.client.login(username='testuser', password='password123')
        response = self.client.get(reverse('user_list'))
        self.assertRedirects(response, reverse('home'))

    def test_admin_user_can_access_user_list(self):
        """
        Verify that an admin user can access the user management area.
        """
        self.client.login(username='adminuser', password='password123')
        response = self.client.get(reverse('user_list'))
        self.assertEqual(response.status_code, 200)
