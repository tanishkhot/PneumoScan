import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# config
MODEL_PATH = 'pneumoscan_efficientnet.pth'
NUM_CLASSES = 2
IMG_SIZE = 224

# loading
@st.cache_resource
def load_model():
    """
    Loads the pre-trained EfficientNetB0 model and modifies its classifier.
    The model is loaded onto the CPU.
    """
    # loading architecture
    model = models.efficientnet_b0(weights=None)
    
    # replace the classifier
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(num_ftrs, NUM_CLASSES)
    )
    
    # load the trained weights
    # using map_location='cpu' to ensure it runs on any machine
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    class_names = checkpoint['class_names']
    
    model.eval() 
    return model, class_names

# --- Grad-CAM implementation ---
class GradCAM:
    """
    Implements Grad-CAM for visualizing model decisions.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # register hooks to the target layer
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, index=None):
        output = self.model(x)
        if index is None:
            index = torch.argmax(output)
        
        # zero out gradients
        self.model.zero_grad()
        # backward pass from the target class
        output[0][index].backward(retain_graph=True)
        
        # get gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # pool the gradients across channels
        pooled_gradients = np.mean(gradients, axis=(1, 2))
        
        # weight the channels by the gradients
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
            
        # average the channels to get the heatmap
        heatmap = np.mean(activations, axis=0)

        # ReLU to keep only positive influences
        heatmap = np.maximum(heatmap, 0)
        
        # ormalize the heatmap
        heatmap /= np.max(heatmap)
        
        return heatmap

def superimpose_heatmap(heatmap, original_image):
    """
    Superimposes the heatmap on the original image.
    """
    # Resize heatmap to match the original image
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose the heatmap
    superimposed_img = heatmap * 0.4 + original_image
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return superimposed_img

# --- Image Preprocessing ---
def preprocess_image(image_bytes):
    """
    Takes image bytes, preprocesses it, and returns a tensor.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_bytes).convert('RGB')
    return transform(image).unsqueeze(0), np.array(image)

# --- Streamlit UI ---
st.set_page_config(
    page_title="PneumoScan",
    page_icon="🫁",
    layout="wide"
)

st.title("🩺 PneumoScan: AI-Powered Pneumonia Detection")
st.write("""
    Hello! My name is Tanish Khot and I built this AI application to help detect pneumonia in chest X-ray images.
         \n\n Warning: These are just predictions and should not be used as a substitute for professional medical advice.
         This can be used by doctors to assist in their diagnosis, but it is not a replacement for human expertise.
         \n For the engineers: this model is trained on the [Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle.
         \bIt uses EfficientNetB0 architecture, which is known for its efficiency and accuracy in image classification tasks.
         The model has been trained to distinguish between normal and pneumonia-affected chest X-rays.
         The Grad-CAM technique is used to provide visual explanations of the model's predictions, highlighting
         the areas of the X-ray that contributed most to the decision.
         """)


st. write("Of course, I don't expect you to have chest X-ray images lying around, so I have provided a sample dataset for you to test the model with.")
st.write("You can find the dataset in the [Drive Folder](https://drive.google.com/drive/folders/11AqtMC4CBRIoYDNOdDGn8FYEqkz2Hwgq?usp=sharing) folder.")
st.write("Upload a chest X-ray image and the AI will analyze it for signs of pneumonia, providing a visual explanation using Grad-CAM.")

# Load the model
with st.spinner('Loading the AI model... This may take a moment.'):
    model, class_names = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess the image
    input_tensor, original_image = preprocess_image(uploaded_file)
    
    # Get model prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_class = class_names[predicted_idx.item()]
        confidence_score = confidence.item()

    # --- Display Results ---
    st.header("Analysis Results")
    col1, col2 = st.columns(2)

    with col1:
        st.image(original_image, caption='Uploaded X-Ray', use_container_width=True)

    with col2:
        if predicted_class == 'PNEUMONIA':
            st.error(f"**Prediction:** {predicted_class}")
        else:
            st.success(f"**Prediction:** {predicted_class}")
        
        st.metric(label="Confidence Score", value=f"{confidence_score:.2%}")
        st.info("The confidence score indicates the model's certainty in its prediction.")

    # --- Grad-CAM Visualization ---
    st.header("Visual Explanation (Grad-CAM)")
    with st.spinner("Generating heatmap..."):
        # Initialize Grad-CAM
        # The target layer is the last convolutional block in EfficientNetB0's features
        grad_cam = GradCAM(model, model.features[-1])
        heatmap = grad_cam(input_tensor)
        
        # Superimpose heatmap on the original image
        grad_cam_image = superimpose_heatmap(heatmap, cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
        
        st.image(grad_cam_image, caption='Grad-CAM Heatmap', use_container_width=True, channels="BGR")
        st.info("""
            **How to interpret this heatmap:**
            The bright red areas are the regions the AI model focused on the most to make its decision. 
            For a 'PNEUMONIA' diagnosis, these areas often highlight lung opacities or consolidations.
        """)
