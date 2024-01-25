import json
import os
import time

import numpy as np
import redis
import settings
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image

# TODO
# Connect to Redis and assign to variable `db``
# Make use of settings.py module to get Redis settings like host, port, etc.
db = redis.Redis(
    host=settings.REDIS_IP,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB_ID
)

# TODO
# Load your ML model and assign to variable `model`
# See https://drive.google.com/file/d/1ADuBSE4z2ZVIdn66YDSwxKv-58U7WEOn/view?usp=sharing
# for more information about how to use this model.
model = ResNet50(include_top=True, weights="imagenet")


def load_image(filename):
    """
    Load image.

    Parameters
    ----------
    filename: str
        Image filename
    """
    img = image.load_img(filename, target_size=(224, 224))
    return image.img_to_array(img)


def predict(image_name):
    """
    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.

    Parameters
    ----------
    image_name : str
        Image filename.

    Returns
    -------
    class_name, pred_probability : tuple(str, float)
        Model predicted class as a string and the corresponding confidence
        score as a number.
    """
    class_name = None
    pred_probability = None

    # Load image
    img_path = os.path.join(settings.UPLOAD_FOLDER, image_name)
    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)
    x_batch = np.expand_dims(x, axis=0)
    x_batch = preprocess_input(x_batch)

    preds = model.predict(x_batch)
    _, class_name, pred_probability = decode_predictions(preds, top=1)[0][0]

    return class_name, np.round(pred_probability, 4)


def classify_process():
    """
    Loop indefinitely asking Redis for new jobs.
    When a new job arrives, takes it from the Redis queue, uses the loaded ML
    model to get predictions and stores the results back in Redis using
    the original job ID so other services can see it was processed and access
    the results.

    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.
    """
    while True:
        filenames = []
        ids = []
        for i in range(settings.BATCH_SIZE):
            queue_value = db.lpop(settings.REDIS_QUEUE)
            if queue_value is None:
                break
            data = json.loads(queue_value.decode())
            filename = data.get("image_name")
            id = data.get("id")
            if not isinstance(filename, type(None)):
                ids.append(id)
                filenames.append(os.path.join(settings.UPLOAD_FOLDER, filename))

        if len(filenames) > 0:
            dataset = np.array([load_image(filename) for filename in filenames])
            dataset_batch = preprocess_input(dataset)
            predictions = model.predict(dataset_batch, batch_size=settings.BATCH_SIZE)
            predictions = decode_predictions(predictions, top=1)
    
            for prediction, job_id in zip(predictions, ids):
                _, class_name, pred_probability = prediction[0]
                pred_probability = np.round(pred_probability, 4)
                result = dict(
                    prediction=class_name,
                    score=float(f"{pred_probability:.4f}"))
                db.set(job_id, json.dumps(result))       

        # Sleep for a bit
        time.sleep(settings.SERVER_SLEEP)


if __name__ == "__main__":
    # Now launch process
    print("Launching ML service...")
    classify_process()
