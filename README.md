🌪️ Cyclone Detection System using Autoencoder & Streamlit

One-Class Deep Learning Model for Satellite-Based Cyclone Identification

This project builds a complete end-to-end cyclone detection pipeline using a Convolutional Autoencoder, classical OpenCV-based cyclone localization, and an interactive Streamlit web application for real-time inference and visualization.

The model detects cyclone regions by analyzing reconstruction error, making it effective even with limited annotated data. It also performs automated cyclone center marking using Hough Lines, Hough Circles, contour analysis, and HSV-based red marker detection.

🚀 Features
🔥 Core Capabilities
Custom-built CNN Autoencoder for one-class cyclone modeling

Real-time detection of cyclone signatures in satellite imagery

Reconstruction-error–based classification (normal vs anomaly)

Multi-stage cyclone-center localization using:

HSV red-cross detection

HoughLinesP for perpendicular-line detection

HoughCircles for circular patterns

Contour-based fallback detection

Fully interactive Streamlit GUI

Downloadable annotated output images

🧠 Model Architecture
The autoencoder consists of:

Encoder: 4 convolutional layers (32 → 64 → 128 → 256 filters) + dense latent space

Latent Dimension: 256

Decoder: Symmetric Conv2DTranspose layers reconstructing 128×128 images

Loss function: MSE
Optimizer: Adam

🛠️ Tech Stack
Component	Technology
Deep Learning	TensorFlow / Keras
Image Processing	OpenCV, NumPy
GUI / Deployment	Streamlit
Others	PIL, ImageDataGenerator

📁 Project Structure
project/
│── cyclone_streamlit.py          # Main script (training + Streamlit app)
│── Data/
│     └── train/
│           └── cyclone/          # Training images
│── cyclone_autoencoder.weights.h5 # Saved model weights (after training)
│── README.md

🔧 Installation
1. Clone the repository
git clone https://github.com/yourusername/cyclone-detection-autoencoder
cd cyclone-detection-autoencoder

2. Install dependencies
pip install -r requirements.txt


Dependencies include:

tensorflow
streamlit
opencv-python-headless
pillow
numpy

🏋️ Training the Autoencoder
Place your cyclone images inside:

Data/train/cyclone/


Then run:

python cyclone_streamlit.py --mode train --data_dir ./Data --epochs 30 --batch_size 32

The script:

Loads images

Trains the autoencoder

Saves weights to cyclone_autoencoder.weights.h5

🌐 Running the Streamlit App
After training, launch the app:

streamlit run cyclone_streamlit.py -- --mode app --model_weights cyclone_autoencoder.weights.h5

The web app provides:

Image upload

Real-time detection

Cyclone center annotation

Reconstruction error display

Download button for annotated output

📊 How Detection Works
Image is resized to 128×128 and fed into the autoencoder.

Model reconstructs the image.

Reconstruction Error is computed:

error = mean((original - reconstructed)^2)

If error ≤ threshold → Cyclone Detected

The system attempts cyclone center marking using:

Red marker detection (HSV mask)

HoughLinesP intersection

HoughCircles (satellite swirl detection)

Contours as fallback

🖼️ Output Example
Annotated cyclone image

Center circle marking

Text label (Cyclone / Not cyclone)

Reconstruction error printed

📥 Download Annotated Image
Streamlit automatically generates a downloadable annotated.png file after detection.
