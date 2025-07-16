# PneumoScan: AI-Powered Pneumonia Detection

**Author:** Tanish Khot

PneumoScan is an interactive web application built to assist in the detection of pneumonia from chest X-ray images. It leverages a deep learning model to classify images as either "Normal" or "Pneumonia" and provides visual explanations for its predictions using Grad-CAM heatmaps.

---

### Key Features

* **AI-Powered Classification:** Utilizes a state-of-the-art EfficientNetB0 model to analyze chest X-rays.
* **Interactive Web UI:** A user-friendly interface built with Streamlit for easy image uploads and analysis.
* **Explainable AI (XAI):** Implements Grad-CAM to generate heatmaps, highlighting the specific regions in the X-ray that influenced the model's decision.
* **Confidence Scoring:** Displays the model's confidence level for each prediction.

---

<!-- ### 🚀 Live Demo & Screenshots

*(Placeholder: You can add a link to your deployed Streamlit Community Cloud app here later)*

![PneumoScan Screenshot](https://i.imgur.com/your-screenshot-url.png)
*(Placeholder: Replace with a real screenshot of your app)*

--- -->

### Tech Stack

* **Backend & Model:** Python 3.10
* **Deep Learning:** PyTorch
* **Web Framework:** Streamlit
* **Image Processing:** OpenCV, Pillow
* **Numerical Operations:** NumPy

---

### Setup and Installation

To run this project locally, please follow these steps:

1.  **Prerequisites:**
    * Python 3.10 or higher (3.13 not supported yet)
    * `pip` package manager

2.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/tanishkhot/PneumoScan.git](https://github.com/tanishkhot/PneumoScan.git)
    cd PneumoScan
    ```

3.  **Create and Activate a Virtual Environment:**
    ```bash
    # Create the environment
    python3 -m venv venv

    # Activate on macOS/Linux
    source venv/bin/activate

    # Activate on Windows
    .\venv\Scripts\activate
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file by running `pip freeze > requirements.txt` in your activated environment.)*

5.  **Download the Dataset:**
    * Download the "Chest X-Ray Images (Pneumonia)" dataset from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
    * Unzip the file and place the `chest_xray` folder inside the root of the project directory. The structure should be `PneumoScan/chest_xray/`.

---

### How to Run

1.  **(Optional) Train the Model:**
    If you wish to train the model from scratch, run the training script. This will generate the `pneumoscan_efficientnet.pth` file.
    ```bash
    python train_model.py
    ```

2.  **Launch the Streamlit App:**
    Make sure the trained model file (`pneumoscan_efficientnet.pth`) is in the root directory and run the following command:
    ```bash
    streamlit run streamlit_app.py
    ```
    Your web browser will automatically open with the application running.

---

### Model Architecture

The core of this project is a deep learning model built using transfer learning.

* **Base Model:** **EfficientNetB0**, pre-trained on the ImageNet dataset.
* **Technique:** Fine-tuning, where the base convolutional layers are frozen and a custom classifier head is trained on the specific task of pneumonia detection.
* **Key Training Features:**
    * **Class Weighting:** To handle the inherent class imbalance in the dataset.
    * **Data Augmentation:** Techniques like random rotations, flips, and crops were used to create a more robust model.
    * **Early Stopping:** To prevent overfitting and save the best model weights based on validation performance.

### Performance

The model was trained on an Apple M1 machine, with the following results:

* **Final Test Accuracy:** **85.42%**
* **Best Validation Accuracy:** **75.00%**
* **Total Training Time:** ~90 minutes
* **Early Stopping:** The training was automatically stopped after 9 epochs (out of a maximum of 25) as the validation loss ceased to improve, preventing overfitting.


---


### Disclaimer

This tool is an academic project and is **not a substitute for professional medical advice**. The predictions made by the AI are intended to assist and supplement, not replace, the diagnosis of a qualified medical professional. Do not use this for self-diagnosis.
