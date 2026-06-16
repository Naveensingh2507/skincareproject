import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import onnxruntime as ort

app = FastAPI(title="AI Skincare Platform ML Backend")

# Enable CORS so your Lovable web app can communicate with this API securely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your specific frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the ONNX Model globally on startup
MODEL_PATH = "skin_type_model.onnx"
try:
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    print(f"🚀 ONNX Model successfully loaded! Expected input field: {input_name}")
except Exception as e:
    print(f"❌ Error loading ONNX model file: {e}")
    session = None

# Class indices matching your training exactly: {'acne': 0, 'dry': 1, 'oil': 2}
CLASSES = ["acne", "dry", "oil"]

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Processes raw image bytes to match MobileNetV2's training data specifications."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((224, 224))
        
        # Convert to a numpy array and scale pixel values to [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Add the batch dimension: shape changes from (224, 224, 3) to (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

@app.post("/predict")
async def predict_skin_type(file: UploadFile = File(...)):
    if session is None:
        raise HTTPException(status_code=500, detail="Inference engine is uninitialized.")
    
    contents = await file.read()
    input_tensor = preprocess_image(contents)
    
    # Run the image tensor through the ONNX model layers
    outputs = session.run(None, {input_name: input_tensor})
    probabilities = outputs[0][0]  # Extract probabilities for the first batch item
    
    # Generate mapping for confidence percentages
    confidence_scores = {CLASSES[i]: round(float(probabilities[i]) * 100, 2) for i in range(len(CLASSES))}
    top_prediction = CLASSES[np.argmax(probabilities)]
    
    return {
        "status": "success",
        "prediction": top_prediction,
        "confidence_scores": confidence_scores
    }

@app.get("/health")
def health_check():
    return {"status": "online", "model_loaded": session is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)