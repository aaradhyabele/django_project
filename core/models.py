from django.db import models

# Create your models here.

class TransactionRecord(models.Model):
    amount = models.FloatField()
    transactions = models.IntegerField()
    hour = models.IntegerField()
    
    # Results
    fraud_result = models.CharField(max_length=20) # "Fraud" or "Normal"
    risk_level = models.CharField(max_length=20)   # "High Risk", etc.
    risk_score = models.FloatField(default=0.0)
    fraud_reason = models.CharField(max_length=255, blank=True, null=True)
    
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.timestamp} - {self.fraud_result}"
