from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import InspectionCase
from .serializers import CaseSerializer

class CaseListCreate(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        qs = InspectionCase.objects.filter(created_by=request.user).order_by("-created_at")
        return Response(CaseSerializer(qs, many=True).data)

    def post(self, request):
        title = request.data.get("title", "").strip()
        case = InspectionCase.objects.create(title=title, created_by=request.user)
        return Response(CaseSerializer(case).data, status=201)
