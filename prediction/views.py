import os, warnings, traceback
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
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.cache import never_cache
from django.conf import settings
from .models import RiceInfo, RiceModel

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
VGG16_PATH = os.path.join(settings.BASE_DIR, 'models', 'best_VGG16_stage2.weights.h5')
MOBILENET_PATH = os.path.join(settings.BASE_DIR, 'models', 'MobileNetV2_rice62_final.weights.h5')
META_MODEL_PATH = os.path.join(settings.BASE_DIR, 'models', 'xgb_meta_model.json')

# Load classes and model from DB
def load_classes_and_model():
    try:
        rice_classes = list(RiceInfo.objects.values_list('variety_name', flat=True))
    except:
        rice_classes = []
    try:
        active_vgg_model = RiceModel.objects.filter(is_active=True, model_type='vgg').first()
    except:
        active_vgg_model = None
    try:
        active_ensemble_model = RiceModel.objects.filter(is_active=True, model_type='ensemble').first()
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

    if ACTIVE_VGG_MODEL and ACTIVE_VGG_MODEL.model_file and os.path.exists(ACTIVE_VGG_MODEL.model_file.path):
        try:
            model.load_weights(ACTIVE_VGG_MODEL.model_file.path)
            print("✅ Loaded VGG16 weights from:", ACTIVE_VGG_MODEL.model_file.path)
        except Exception as e:
            print(f"[ERROR] Failed to load VGG16 weights: {e}")
    else:
        print("[ERROR] No active VGG16 model or file not found")

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

# -----------------------------
# Prediction View
# -----------------------------
@csrf_exempt
@never_cache
def predict(request):
    warnings.filterwarnings("ignore", category=UserWarning)

    if request.method == "POST":
        try:
            image_file = request.FILES.get("rice_image")
            if not image_file:
                return JsonResponse({"error": "No image provided"}, status=400)

            # Preprocess
            image = Image.open(image_file).convert("RGB")

            # Save the uploaded image to fixed location
            path = os.path.join(settings.MEDIA_ROOT, 'predictions', 'current.jpg')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            image.save(path)

            image = image.resize(IMAGE_SIZE)
            image_array = np.expand_dims(np.array(image, dtype=np.float32), axis=0)

            model_type = request.POST.get('model_type', 'vgg')

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
                confidence = pred_prob * 100
            else:
                return JsonResponse({"error": "Invalid model_type. Use 'vgg' or 'ensemble'"}, status=400)

            rice_info_obj = RiceInfo.objects.filter(variety_name=predicted_class).first()
            rice_info = rice_info_obj.info if rice_info_obj else "No info available."

            # Close the image
            image.close()

            # Delete the uploaded image file if it's a temporary file
            if hasattr(image_file, 'temporary_file_path'):
                try:
                    os.remove(image_file.temporary_file_path())
                except OSError:
                    pass  # Ignore if deletion fails

            return JsonResponse({
                "predicted_variety": predicted_class,
                "confidence": confidence,
                "rice_info": rice_info,
                "model_type": model_type,
                "message": f"Predicted Rice Variety: {predicted_class} ({confidence:.2f}% confidence)"
            })

        except Exception as e:
            traceback.print_exc()
            return JsonResponse({"error": str(e)}, status=500)

    return render(request, "prediction/predict.html")

def home(request):
    return render(request, "prediction/home.html")

# New endpoint to get active model
def get_model(request):
    active_model = RiceModel.objects.filter(is_active=True).first()
    if not active_model or not active_model.tflite_file or not os.path.exists(active_model.tflite_file.path):
        return JsonResponse({"error": "No active model available"}, status=404)
    try:
        with open(active_model.tflite_file.path, 'rb') as f:
            response = HttpResponse(f.read(), content_type='application/octet-stream')
            response['Content-Disposition'] = f'attachment; filename="{active_model.name}.tflite"'
            return response
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

# New endpoint to get rice info
def get_rice_info(request):
    rice_infos = list(RiceInfo.objects.values('variety_name', 'info', 'updated_at'))
    return JsonResponse({"rice_infos": rice_infos})
