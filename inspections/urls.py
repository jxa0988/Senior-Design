from django.urls import path
from .views import CaseListCreate

urlpatterns = [
    path("cases/", CaseListCreate.as_view()),
]
