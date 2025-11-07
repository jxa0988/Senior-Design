# Minimal models (InspectionCase, ImageAsset, InferenceJob)
from django.conf import settings
from django.db import models

class InspectionCase(models.Model):
    title = models.CharField(max_length=200)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

class ImageAsset(models.Model):
    case = models.ForeignKey(InspectionCase, on_delete=models.CASCADE, related_name="images")
    path = models.CharField(max_length=512)                     # local path or gs://... (later)
    status = models.CharField(max_length=32, default="pending") # pending|uploaded|inferred|failed
    uploaded_at = models.DateTimeField(null=True, blank=True)

class InferenceJob(models.Model):
    image = models.OneToOneField(ImageAsset, on_delete=models.CASCADE, related_name="job")
    status = models.CharField(max_length=32, default="queued")  # queued|running|succeeded|failed
    predictions = models.JSONField(null=True, blank=True)
    error = models.TextField(blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

