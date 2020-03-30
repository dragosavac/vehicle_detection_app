from rest_framework import serializers


class InferenceSerializer(serializers.Serializer):
    date = serializers.DateTimeField(required=True, allow_null=False)
    image = serializers.FileField(required=True, allow_null=False, allow_empty_file=False)
    value = serializers.CharField(read_only=True, allow_null=True)

    def update(self, instance, validated_data):
        pass

    def create(self, validated_data):
        pass


class DateTimeRangeSerializer(serializers.Serializer):
    start_date = serializers.DateTimeField(required=True, allow_null=False)
    end_date = serializers.DateTimeField(required=True, allow_null=False)

    def update(self, instance, validated_data):
        pass

    def create(self, validated_data):
        pass
