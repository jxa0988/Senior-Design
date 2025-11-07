from rest_framework import serializers
from .models import InspectionCase

class CaseSerializer(serializers.ModelSerializer):
    class Meta:
        model = InspectionCase
        fields = ["id", "title", "created_at"]
