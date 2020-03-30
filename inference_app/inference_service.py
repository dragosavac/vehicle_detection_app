import json
from datetime import datetime

from PIL import Image
from rest_framework.exceptions import ValidationError

from infer_model.infer import run_inference, DEFAULT_LABELMAP_PATH
from .models import InferenceInstance
from utilities.predict_vehicle_lane import predict_vehicle_lane
from utilities.constants import *
import numpy as np
from collections import Counter


class InferenceService:

    @classmethod
    def run_inference(cls, image: Image, date: datetime) -> dict:
        image = Image.open(image)
        if image.format.lower() not in ALLOWED_FILE_FORMATS:
            raise ValidationError()
        inference_result: str = run_inference(
            image=image,
            model_path=DEFAULT_MODEL_PATH,
            labelmap_path=DEFAULT_LABELMAP_PATH,
            threshold=DEFAULT_THRESHOLD,
            output_path=None
        )

        model_output = json.loads(inference_result)
        vehicles_in_lane_list = predict_vehicle_lane(model_output, angles, distances)
        vehicles_in_lane_dict = Counter(vehicles_in_lane_list)
        vehicles_in_lane_order_dict = {index: vehicles_in_lane_dict[index] for index in range(10)}

        cls._save_inference(inference_result=vehicles_in_lane_order_dict, date=date)

        final_output = {}
        final_output['trucks per lane'] = vehicles_in_lane_order_dict
        final_output['date'] = date

        return final_output

    @classmethod
    def get_stats_for_datetime_range(cls, start_date: datetime, end_date: datetime) -> dict:
        inference_instances = InferenceInstance.objects.filter(calculation_datetime__gt=start_date,
                                                               calculation_datetime__lt=end_date).all()
        data = np.ndarray(shape=(len(inference_instances), 10))
        for i, instance in enumerate(inference_instances):
            l = instance.value.split(',')
            values = [int(item.strip("}").split(": ")[1]) for item in l]
            data[i] = np.array(values)
        maximum = np.amax(data, axis=0)
        minimum = np.amax(data, axis=0)
        mean = np.mean(data, axis=0)
        median = np.median(data, axis=0)
        std = np.std(data, axis=0)
        stats_per_lane_dict = {
            'maximum': maximum,
            'minimum': minimum,
            'median': median,
            'mean': mean,
            'standard_deviation': std
        }
        return stats_per_lane_dict

    @classmethod
    def _save_inference(cls, inference_result: dict, date: datetime):
        inference_instance = InferenceInstance(value=inference_result, calculation_datetime=date)
        inference_instance.save()

