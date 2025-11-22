
import argparse
import os
from pathlib import Path
from io import BytesIO
import math
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator


# Model builder

def build_autoencoder(input_shape=(128, 128, 3), latent_dim=256):
    inp = layers.Input(shape=input_shape, name='input_layer')
    # Encoder
    x = layers.Conv2D(32, 3, activation='relu', padding='same', strides=2)(inp)  # 64x64
    x = layers.Conv2D(64, 3, activation='relu', padding='same', strides=2)(x)    # 32x32
    x = layers.Conv2D(128, 3, activation='relu', padding='same', strides=2)(x)   # 16x16
    x = layers.Conv2D(256, 3, activation='relu', padding='same', strides=2)(x)   # 8x8
    x = layers.Flatten()(x)
    encoded = layers.Dense(latent_dim, activation='relu', name='encoded')(x)

    # Decoder
    x = layers.Dense(8 * 8 * 256, activation='relu')(encoded)
    x = layers.Reshape((8, 8, 256))(x)
    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(x)  # 16x16
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)   # 32x32
    x = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)   # 64x64
    x = layers.Conv2DTranspose(16, 3, strides=2, padding='same', activation='relu')(x)   # 128x128
    decoded = layers.Conv2D(3, 3, activation='sigmoid', padding='same', name='decoded')(x)

    autoencoder = models.Model(inputs=inp, outputs=decoded, name='conv_autoencoder')
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder



# Training

def train_autoencoder(data_dir,
                      model_weights_path='cyclone_autoencoder.weights.h5',
                      img_size=(128, 128),
                      batch_size=32,
                      epochs=30,
                      val_split=0.1):
    data_dir = Path(data_dir)
    train_dir = data_dir / 'train'
    cyclone_dir = train_dir / 'cyclone'
    if not cyclone_dir.exists():
        raise FileNotFoundError(f"Expected cyclone images in: {cyclone_dir}\nPlease create {cyclone_dir} and put images there.")

    datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=val_split)

    # class_mode='input' -> generator yields (x, x) pairs suitable for autoencoder training
    train_gen = datagen.flow_from_directory(
        directory=str(train_dir),
        classes=['cyclone'],
        target_size=img_size,
        batch_size=batch_size,
        class_mode='input',
        subset='training',
        shuffle=True
    )
    val_gen = datagen.flow_from_directory(
        directory=str(train_dir),
        classes=['cyclone'],
        target_size=img_size,
        batch_size=batch_size,
        class_mode='input',
        subset='validation',
        shuffle=False
    )

    print(f"Found train samples: {train_gen.samples}, val samples: {val_gen.samples}")
    if train_gen.samples == 0:
        raise RuntimeError("No training images found. Check Data/train/cyclone contains images.")

    autoencoder = build_autoencoder(input_shape=(img_size[0], img_size[1], 3))

    steps_per_epoch = max(1, math.ceil(train_gen.samples / batch_size))
    validation_steps = max(1, math.ceil(val_gen.samples / batch_size))

    print(f"Training steps_per_epoch={steps_per_epoch}, validation_steps={validation_steps}")

    autoencoder.fit(
        train_gen,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps
    )

    # Save only weights — model architecture is always rebuilt from code on load
    autoencoder.save_weights(model_weights_path)
    print(f"Weights saved to {model_weights_path}")
    return autoencoder



# Predict + annotate (heuristic red circle)

def reconstruction_error(original, reconstructed):
    return np.mean((original.astype('float32') - reconstructed.astype('float32')) ** 2)


def _find_red_plus_center(orig_rgb, min_area=50, debug=False):
    
    import math
    img = orig_rgb.copy()
    h, w = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # two red ranges in HSV (low and high hue)
    lower1 = np.array([0, 80, 50])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 80, 50])
    upper2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Try HoughLinesP to detect plus (two crossing lines)
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=30, minLineLength=min(w, h)//10, maxLineGap=10)

    if lines is not None and len(lines) >= 2:
        # convert lines to (x1,y1,x2,y2)
        # find two lines that are roughly perpendicular and compute intersection
        best = None
        for i in range(len(lines)):
            for j in range(i+1, len(lines)):
                x1,y1,x2,y2 = lines[i][0]
                x3,y3,x4,y4 = lines[j][0]
                # compute angle of each
                ang1 = math.atan2(y2-y1, x2-x1)
                ang2 = math.atan2(y4-y3, x4-x3)
                diff = abs(abs(ang1-ang2) - math.pi/2)
                # accept if roughly perpendicular (tolerance ~20 degrees)
                if diff < (20 * math.pi/180):
                    # compute intersection
                    def intersect(a1,a2,b1,b2):
                        # a1=(x1,y1), a2=(x2,y2) ...
                        x1,y1 = a1; x2,y2 = a2; x3,y3 = b1; x4,y4 = b2
                        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
                        if denom == 0:
                            return None
                        px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
                        py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
                        return int(px), int(py)
                    pt = intersect((x1,y1),(x2,y2),(x3,y3),(x4,y4))
                    if pt is not None:
                        cx, cy = pt
                        # keep only if inside image bounds
                        if 0 <= cx < w and 0 <= cy < h:
                            return (cx, cy)
        # if no perpendicular pair found, continue to contour method

    # fallback: largest red contour centroid
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area >= min_area:
            M = cv2.moments(largest)
            if M.get('m00', 0) != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                return (cx, cy)
    return None


def predict_and_annotate_autoencoder(pil_image, autoencoder, img_size=(128, 128), threshold=0.01):
   
    orig_rgb = np.array(pil_image.convert('RGB'))
    h, w = orig_rgb.shape[:2]

    # Prepare input for autoencoder
    inp = cv2.resize(orig_rgb, img_size)
    inp_norm = inp.astype('float32') / 255.0
    inp_batch = np.expand_dims(inp_norm, axis=0)

    reconstructed = autoencoder.predict(inp_batch)[0]  # [0..1]
    err = reconstruction_error(inp_norm, reconstructed)
    is_cyclone = err <= threshold

    annotated_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)

    # 1) Try red '+' detection first (most reliable for your dataset)
    red_center = _find_red_plus_center(orig_rgb, min_area=30)
    if red_center is not None:
        x, y = red_center
        # radius relative to image
        r = int(min(w, h) * 0.12)
        cv2.circle(annotated_bgr, (x, y), r, (0, 0, 255), 4)
        cv2.putText(annotated_bgr, f"Cyclone ( Detected center !!)", (max(10, x - r), max(20, y - r - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(annotated_rgb), err, True

    # 2) If no red marker found, keep previous behavior:
    if not is_cyclone:
        cv2.putText(annotated_bgr, f"Not cyclone (err={err:.4f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(annotated_rgb), err, False

    # Cyclone predicted — attempt Hough circles on grayscale
    gray = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)

    circles = None
    try:
        circles = cv2.HoughCircles(gray_blur,
                                   cv2.HOUGH_GRADIENT,
                                   dp=1.2,
                                   minDist=50,
                                   param1=50,
                                   param2=30,
                                   minRadius=10,
                                   maxRadius=0)
    except Exception:
        circles = None

    if circles is not None:
        circles = np.uint16(np.around(circles))
        largest = max(circles[0, :], key=lambda c: c[2])
        x, y, r = int(largest[0]), int(largest[1]), int(largest[2])
        cv2.circle(annotated_bgr, (x, y), r, (0, 0, 255), 4)
        cv2.putText(annotated_bgr, f"Cyclone (err={err:.4f})", (max(10, x - r), max(20, y - r - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(annotated_rgb), err, True

    # Fallback: threshold + largest contour
    _, th = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_cnt = max(contours, key=cv2.contourArea)
        (x, y), r = cv2.minEnclosingCircle(largest_cnt)
        x, y, r = int(x), int(y), int(max(10, int(r * 1.1)))
        cv2.circle(annotated_bgr, (x, y), r, (0, 0, 255), 4)
        cv2.putText(annotated_bgr, f"Cyclone (err={err:.4f})", (max(10, x - r), max(20, y - r - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(annotated_rgb), err, True

    # Last resort: center circle
    cx, cy = w // 2, h // 2
    r = int(min(w, h) * 0.25)
    cv2.circle(annotated_bgr, (cx, cy), r, (0, 0, 255), 4)
    cv2.putText(annotated_bgr, f"Cyclone (err={err:.4f})", (max(10, cx - r), max(20, cy - r - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated_rgb), err, True



# Streamlit app (loads architecture + weights)

def run_streamlit_app(model_weights_path='cyclone_autoencoder.weights.h5'):
    import streamlit as st
    st.set_page_config(page_title='Cyclone One-Class Detector', layout='centered')
    st.title('Cyclone One-Class Detector (Autoencoder)')

    if not os.path.exists(model_weights_path):
        st.warning(f"Weights not found at {model_weights_path}. Train the model first or provide correct path.")
        st.info('To train: run this script with --mode train')
        st.stop()

    @st.cache_resource
    def load_autoencoder(weights_path):
        model = build_autoencoder(input_shape=(128, 128, 3))
        # load_weights expects the same architecture
        model.load_weights(weights_path)
        return model

    autoencoder = load_autoencoder(model_weights_path)

    st.sidebar.header('Options')
    threshold = st.sidebar.slider('Reconstruction error threshold (lower=stricter)', 0.0001, 0.05, 0.01, step=0.0001)
    show_reconstruction = st.sidebar.checkbox('Show reconstruction (debug)', value=False)

    uploaded = st.file_uploader('Upload satellite image', type=['png', 'jpg', 'jpeg', 'tif'])
    if uploaded is None:
        st.info('Upload an image to detect and annotate cyclone.')
        return

    img = Image.open(uploaded).convert('RGB')
    st.subheader('Original image')
    st.image(img, use_container_width=True)


    with st.spinner('Analyzing...'):
        annotated, err, is_cyclone = predict_and_annotate_autoencoder(img, autoencoder, img_size=(128, 128), threshold=threshold)

        if show_reconstruction:
            inp = cv2.resize(np.array(img), (128, 128)).astype('float32') / 255.0
            recon = autoencoder.predict(np.expand_dims(inp, 0))[0]
            recon_img = (recon * 255).astype('uint8')
            recon_img = cv2.resize(recon_img, (img.width, img.height))
            st.subheader('Reconstruction (resized)')
            st.image(Image.fromarray(recon_img), use_container_width=True)

    st.subheader('Result')
    st.image(annotated, use_container_width=True)

    st.markdown(f"**Reconstruction error:** {err:.6f}")
    st.markdown(f"**Detected as cyclone:** {'Yes' if is_cyclone else 'No'}")

    buf = BytesIO()
    annotated.save(buf, format='PNG')
    st.download_button('Download annotated image', data=buf.getvalue(), file_name='annotated.png', mime='image/png')



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', type=str, choices=['train', 'app'], default='app')
    p.add_argument('--data_dir', type=str, default='/mnt/data/Data')
    p.add_argument('--model_weights', type=str, default='cyclone_autoencoder.weights.h5')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=32)
    return p.parse_known_args()


if __name__ == '__main__':
    args, _ = parse_args()
    if args.mode == 'train':
        train_autoencoder(args.data_dir, model_weights_path=args.model_weights, epochs=args.epochs, batch_size=args.batch_size)
    else:
        try:
            run_streamlit_app(model_weights_path=args.model_weights)
        except Exception as e:
            print('To run the app with Streamlit:')
            print('  streamlit run cyclone_streamlit.py -- --mode app --model_weights cyclone_autoencoder.weights.h5')
            raise
