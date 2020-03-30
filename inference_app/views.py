import json

from django.http import HttpResponseBadRequest
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView

from inference_app.serializers import InferenceSerializer, DateTimeRangeSerializer
from inference_app.inference_service import InferenceService


@api_view()
def health_check(request):
    return Response({'System is alive and well'})


class InferenceView(APIView):
    http_method_names = ['post']

    @classmethod
    def post(cls, request, *args, **kwargs):
        serializer = InferenceSerializer(data=request.data)
        if serializer.is_valid():
            inference_result: dict = InferenceService.run_inference(
                image=serializer.validated_data['image'],
                date=serializer.validated_data['date'])
        else:
            return HttpResponseBadRequest('{"error": "`image` and `date` fields are required!"}')

        return Response(inference_result, status=200)


class StatsView(APIView):
    http_method_names = ['get']

    @classmethod
    def get(cls, request, *args, **kwargs):
        serializer = DateTimeRangeSerializer(data=request.query_params)
        if serializer.is_valid():
            stats_result: dict = InferenceService.get_stats_for_datetime_range(
                start_date=serializer.validated_data['start_date'],
                end_date=serializer.validated_data['end_date'])
        else:
            return HttpResponseBadRequest('{"error": `start_date` and `end_date` params are required and '
                                          'format is: YYYY-MM-DDTHH:MM[:SS[.fffff]]}')

        return Response(stats_result, status=200)

