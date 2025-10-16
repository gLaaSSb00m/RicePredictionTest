import os
import sys
import warnings
import traceback
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from xgboost import XGBClassifier

# Add Django project to path
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rice_prediction.settings')

import django
django.setup()

from prediction.models import RiceInfo, RiceModel

# -----------------------------
# Strategy (GPU/CPU)
# -----------------------------
def get_strategy():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        strat = tf.distribute.MirroredStrategy()
        print(f"✅ Using MirroredStrategy on {len(gpus)} GPU(s).")
        return strat
    print("✅ Using default strategy (CPU).")
    return tf.distribute.get_strategy()

strategy = get_strategy()
print("Replicas:", strategy.num_replicas_in_sync)

# -----------------------------
# Config
# -----------------------------
IMAGE_SIZE = (224, 224)
VGG16_PATH = os.path.join('models', 'best_VGG16_stage2.weights.h5')
MOBILENET_PATH = os.path.join('models', 'MobileNetV2_rice62_final.weights.h5')
META_MODEL_PATH = os.path.join('models', 'xgb_meta_model.json')

# Load classes and model from DB
def load_classes_and_model():
    try:
        rice_classes = list(RiceInfo.objects.values_list('variety_name', flat=True))
    except:
        rice_classes = []
    try:
        active_vgg_model = RiceModel.objects.filter(is_active=True, name='VGG16 Model').first()
    except:
        active_vgg_model = None
    try:
        active_ensemble_model = RiceModel.objects.filter(is_active=True, name='Ensemble Model').first()
    except:
        active_ensemble_model = None
    return rice_classes, active_vgg_model, active_ensemble_model

RICE_CLASSES, ACTIVE_VGG_MODEL, ACTIVE_ENSEMBLE_MODEL = load_classes_and_model()

# -----------------------------
# Load XGBoost meta-model
# -----------------------------
xgb_meta = XGBClassifier()
xgb_meta.load_model(META_MODEL_PATH)

# -----------------------------
# Build + Load models
# -----------------------------
with strategy.scope():
    def build_model(num_classes, l2_weight=1e-4, dropout_rate=0.3):
        base_model = VGG16(include_top=False, input_shape=IMAGE_SIZE + (3,), weights="imagenet")
        x = base_model.output
        x = GlobalAveragePooling2D(name="gap")(x)
        x = Dropout(dropout_rate, name="dropout")(x)
        x = Dense(256, activation="relu", kernel_regularizer=l2(l2_weight), name="dense_256")(x)
        x = BatchNormalization(name="bn")(x)
        outputs = Dense(num_classes, activation="softmax", dtype="float32", name="pred")(x)
        return keras.Model(inputs=base_model.input, outputs=outputs, name="VGG16_rice62")

    def build_vgg16_feature_extractor():
        base = VGG16(weights=None, include_top=False, input_shape=(224,224,3))
        x = GlobalAveragePooling2D()(base.output)
        model = keras.Model(inputs=base.input, outputs=x)
        model.load_weights(VGG16_PATH)
        return model

    def build_mobilenetv2_feature_extractor():
        base = MobileNetV2(weights=None, include_top=False, input_shape=(224,224,3))
        x = GlobalAveragePooling2D()(base.output)
        model = keras.Model(inputs=base.input, outputs=x)
        model.load_weights(MOBILENET_PATH)
        return model

    model = build_model(len(RICE_CLASSES))
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])

    vgg_extractor = build_vgg16_feature_extractor()
    mobilenet_extractor = build_mobilenetv2_feature_extractor()

    if ACTIVE_VGG_MODEL and ACTIVE_VGG_MODEL.is_active:
        if ACTIVE_VGG_MODEL.model_file and os.path.exists(ACTIVE_VGG_MODEL.model_file.path):
            try:
                model.load_weights(ACTIVE_VGG_MODEL.model_file.path)
                print("✅ Loaded VGG16 weights from:", ACTIVE_VGG_MODEL.model_file.path)
            except Exception as e:
                print(f"[ERROR] Failed to load VGG16 weights from DB path: {e}")
                # Fallback to hardcoded path
                if os.path.exists(VGG16_PATH):
                    try:
                        model.load_weights(VGG16_PATH)
                        print("✅ Loaded VGG16 weights from fallback path:", VGG16_PATH)
                    except Exception as e2:
                        print(f"[ERROR] Failed to load VGG16 weights from fallback: {e2}")
                else:
                    print(f"[ERROR] Fallback VGG16 weights file not found at {VGG16_PATH}")
        else:
            # No DB file, try hardcoded
            if os.path.exists(VGG16_PATH):
                try:
                    model.load_weights(VGG16_PATH)
                    print("✅ Loaded VGG16 weights from hardcoded path:", VGG16_PATH)
                except Exception as e:
                    print(f"[ERROR] Failed to load VGG16 weights from hardcoded: {e}")
            else:
                print(f"[ERROR] No VGG16 weights file found at {VGG16_PATH}")
    else:
        print("[INFO] No active VGG16 model")

    if ACTIVE_ENSEMBLE_MODEL:
        if ACTIVE_ENSEMBLE_MODEL.vgg_weights_file and os.path.exists(ACTIVE_ENSEMBLE_MODEL.vgg_weights_file.path):
            try:
                vgg_extractor.load_weights(ACTIVE_ENSEMBLE_MODEL.vgg_weights_file.path)
                print("✅ Loaded Ensemble VGG16 weights from:", ACTIVE_ENSEMBLE_MODEL.vgg_weights_file.path)
            except Exception as e:
                print(f"[ERROR] Failed to load Ensemble VGG16 weights: {e}")
        else:
            print("[ERROR] No active Ensemble VGG16 weights file")

        if ACTIVE_ENSEMBLE_MODEL.mobilenet_weights_file and os.path.exists(ACTIVE_ENSEMBLE_MODEL.mobilenet_weights_file.path):
            try:
                mobilenet_extractor.load_weights(ACTIVE_ENSEMBLE_MODEL.mobilenet_weights_file.path)
                print("✅ Loaded Ensemble MobileNetV2 weights from:", ACTIVE_ENSEMBLE_MODEL.mobilenet_weights_file.path)
            except Exception as e:
                print(f"[ERROR] Failed to load Ensemble MobileNetV2 weights: {e}")
        else:
            print("[ERROR] No active Ensemble MobileNetV2 weights file")

        if ACTIVE_ENSEMBLE_MODEL.xgb_model_file and os.path.exists(ACTIVE_ENSEMBLE_MODEL.xgb_model_file.path):
            try:
                xgb_meta.load_model(ACTIVE_ENSEMBLE_MODEL.xgb_model_file.path)
                print("✅ Loaded XGBoost model from:", ACTIVE_ENSEMBLE_MODEL.xgb_model_file.path)
            except Exception as e:
                print(f"[ERROR] Failed to load XGBoost model: {e}")
        else:
            print("[ERROR] No active XGBoost model file")
    else:
        print("[INFO] No active Ensemble model")

# Helper function for feature extraction
def extract_features(img_array, model, preprocess_func):
    img_array = preprocess_func(img_array)
    feat = model.predict(img_array, verbose=0)
    return feat.flatten()

def predict_image(image_path, model_type='vgg'):
    warnings.filterwarnings("ignore", category=UserWarning)

    try:
        if not os.path.exists(image_path):
            print(f"Error: Image file '{image_path}' not found.")
            return

        # Preprocess
        image = Image.open(image_path).convert("RGB")
        image = image.resize(IMAGE_SIZE)
        image_array = np.expand_dims(np.array(image, dtype=np.float32), axis=0)

        if model_type == 'vgg':
            # Original VGG16 prediction
            image_array_vgg = image_array / 255.0
            preds = model.predict(image_array_vgg, verbose=0)
            idx = int(np.argmax(preds[0]))
            predicted_class = RICE_CLASSES[idx]
            confidence = float(np.max(preds[0]) * 100)
        elif model_type == 'ensemble':
            # Ensemble prediction
            feat_vgg = extract_features(image_array.copy(), vgg_extractor, vgg_preprocess)
            feat_mobile = extract_features(image_array.copy(), mobilenet_extractor, mobilenet_preprocess)
            stacked_feat = np.hstack([feat_vgg, feat_mobile]).reshape(1, -1)
            pred_index = xgb_meta.predict(stacked_feat)[0]
            pred_prob = xgb_meta.predict_proba(stacked_feat).max()
            predicted_class = RICE_CLASSES[pred_index]
            confidence = float(pred_prob * 100)
        else:
            print("Error: Invalid model_type. Use 'vgg' or 'ensemble'")
            return

        rice_info_obj = RiceInfo.objects.filter(variety_name=predicted_class).first()
        rice_info = rice_info_obj.info if rice_info_obj else "No info available."

        # Close the image
        image.close()

        print(f"Predicted Rice Variety: {predicted_class}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"Model Type: {model_type}")
        print(f"Rice Info: {rice_info}")

    except Exception as e:
        traceback.print_exc()
        print(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict rice variety from image using shell.")
    parser.add_argument("image_path", help="Path to the rice image file")
    parser.add_argument("--model_type", choices=['vgg', 'ensemble'], default='vgg', help="Model type to use for prediction (default: vgg)")

    args = parser.parse_args()

    predict_image(args.image_path, args.model_type)
