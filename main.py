from fastapi import FastAPI, UploadFile
import uvicorn
import numpy as np
import tensorflow as tf
import cv2
from fastapi.responses import JSONResponse
from io import BytesIO

app = FastAPI()

model = tf.keras.models.load_model("best_balanced_model.keras")


# Find last Conv2D layer automatically

last_conv_layer_name = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer_name = layer.name
        break

if last_conv_layer_name is None:
    raise ValueError("No Conv2D layer found!")



# Grad-CAM Function

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    layer_idx = None
    for idx, layer in enumerate(model.layers):
        if layer.name == last_conv_layer_name:
            layer_idx = idx
            break

    conv_model = tf.keras.Model(
        inputs=model.inputs[0] if isinstance(model.inputs, list) else model.inputs,
        outputs=model.layers[layer_idx].output
    )

    classifier_input = tf.keras.Input(shape=model.layers[layer_idx].output.shape[1:])
    x = classifier_input
    for layer in model.layers[layer_idx + 1:]:
        x = layer(x)
    classifier_model = tf.keras.Model(inputs=classifier_input, outputs=x)

    with tf.GradientTape() as tape:
        conv_output = conv_model(img_array)
        tape.watch(conv_output)
        predictions = classifier_model(conv_output)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0].numpy()
    pooled_grads = pooled_grads.numpy()

    for i in range(len(pooled_grads)):
        conv_output[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_output, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1

    return heatmap

# Overlay heatmap

def create_heatmap_image(original_img, heatmap):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(heatmap_color, 0.4, original_img, 0.6, 0)
    _, buffer = cv2.imencode(".png", overlay)
    return buffer.tobytes()



# API Endpoint

@app.post("/predict")
async def predict(file: UploadFile):

    content = await file.read()
    img_array = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    img_resized = cv2.resize(img, (150, 150))
    img_norm = img_resized.astype("float32") / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    prob = model.predict(img_input)[0][0]
    pred_label = "Tumor" if prob > 0.5 else "No Tumor"

    heatmap = make_gradcam_heatmap(img_input, model, last_conv_layer_name)
    heatmap_img = create_heatmap_image(img_resized, heatmap)

    return JSONResponse({
        "prediction": pred_label,
        "confidence": float(prob),
        "heatmap": list(heatmap_img)
    })



# Run the API

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
