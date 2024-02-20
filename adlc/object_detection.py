from keras.models import load_model
from keras_cv.layers import MultiClassNonMaxSuppression,Resizing
from keras_cv import bounding_box
import tensorflow as tf

class TargetDetector:
    def __init__(self) -> None:
        self.model = load_model("./model/adlc-detect_1703302468.3469934.keras")

    def find_targets(self, images):
        # Convert list of pillow images to tensors
        tensor_images = tf.convert_to_tensor([[tf.convert_to_tensor(img)] for img in images])

        # Make predictions and convert to ragged (variable length) tensors 
        encoded_predictions = self.model(tensor_images)
        y_pred = self.model.decode_predictions(encoded_predictions, tensor_images)
        return y_pred