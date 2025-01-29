from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from glob import glob
import joblib

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
app.config['DETECTED_FOLDER'] = 'detected_images'  
app.config['REMOVED_FOLDER'] = 'removed_shadows'

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)
os.makedirs(app.config['DETECTED_FOLDER'], exist_ok=True)
os.makedirs(app.config['REMOVED_FOLDER'], exist_ok=True) 

# Initialize model
model = None

# Feature extraction and model training functions 
def extract_features(image):
    h, w, _ = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform').flatten()

    gray_flat = gray.flatten()
    features = np.hstack([gray_flat[:, None], lbp[:, None]])
    return features

def train_model():
    image_paths = glob('ISTD_Dataset/train/train_A/*')
    mask_paths = glob('ISTD_Dataset/train/train_B/*')
    X, Y = load_dataset(image_paths, mask_paths)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=60, random_state=42, verbose=1)
    clf.fit(X_train, Y_train)

    Y_pred = clf.predict(X_test)

    print("Classification Report:")
    print(classification_report(Y_test, Y_pred, zero_division=0))
    print("Accuracy:", accuracy_score(Y_test, Y_pred))

    joblib.dump(clf, 'shadow_detection_model.pkl')
    print("Model saved as 'shadow_detection_model.pkl'")

    return clf

def load_dataset(image_paths, mask_paths, target_size=(128, 128), max_images=150):
    X, Y = [], []
    image_paths = image_paths[:max_images]
    mask_paths = mask_paths[:max_images]

    for img_path, mask_path in zip(image_paths, mask_paths):
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, target_size)
        mask = cv2.resize(mask, target_size)

        features = extract_features(img)
        labels = mask.flatten()

        X.append(features)
        Y.append(labels)

    X = np.vstack(X)
    Y = np.hstack(Y)
    
    return X, Y

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        static_path = os.path.join(app.config['STATIC_FOLDER'], 'original_image.jpg')
        file.seek(0) 
        file.save(static_path)

        return render_template('index.html', original_image=file.filename)


@app.route('/detect_shadow/<filename>')
def detect_shadow(filename):
    global model
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if model is None:
        if os.path.exists('shadow_detection_model.pkl'):
            model = joblib.load('shadow_detection_model.pkl')
        else:
            model = train_model()
    shadow_img, _ = detect_and_highlight_shadow(filepath, model)
    
    # Save the detected image to the new folder
    detected_image_path = os.path.join(app.config['DETECTED_FOLDER'], f'detected_{filename}')
    cv2.imwrite(detected_image_path, shadow_img)
    
    output_path = os.path.join(app.config['STATIC_FOLDER'], 'processed_shadow_image.jpg')
    cv2.imwrite(output_path, shadow_img)
    
    return render_template('index.html', original_image=filename, processed_image='processed_shadow_image.jpg')

@app.route('/remove_shadow/<filename>')
def remove_shadow(filename):

    original_image_path = os.path.join(app.config['STATIC_FOLDER'], 'original_image.jpg')
    original_image = cv2.imread(original_image_path)
    
    if original_image is None:
        return "Error: Unable to load original image.", 400
    
    # Load the thresholded mask (created during detection)
    thresholded_mask_path = os.path.join(app.config['STATIC_FOLDER'], 'shadow_thresholded_image.jpg')
    thresholded_mask = cv2.imread(thresholded_mask_path, cv2.IMREAD_GRAYSCALE)
    
    if thresholded_mask is None:
        return "Error: Unable to load thresholded mask image.", 400
    
    # Resize the thresholded mask to match the original image dimensions
    thresholded_mask_resized = cv2.resize(thresholded_mask, (original_image.shape[1], original_image.shape[0]))
    
    # Remove the shadow using the original image and the thresholded mask
    shadow_removed_img = remove_shadow_from_image(original_image, thresholded_mask_resized)
    
    if shadow_removed_img is None:
        return "Error: Unable to process the image.", 400
    
    # Save the shadow-removed image to the new folder
    static_removed_image_filename = f'removed_processed_shadow_image.jpg'
    removed_image_filename = f'removed_{filename}'

    static_image_path = os.path.join(app.config['STATIC_FOLDER'], static_removed_image_filename)
    removed_image_path = os.path.join(app.config['REMOVED_FOLDER'], removed_image_filename)

    cv2.imwrite(static_image_path, shadow_removed_img)
    cv2.imwrite(removed_image_path, shadow_removed_img)
    
    # Render the template with the shadow-removed image
    return render_template('index.html', original_image='original_image.jpg', processed_image=static_removed_image_filename)

@app.route('/download/<filename>')
def download_image(filename):
    # Ensure the filename is safe and points to the correct folder
    if filename.startswith('removed_'):
        return send_from_directory(app.config['REMOVED_FOLDER'], filename, as_attachment=True)
    else:
        return "File not found.", 404

def detect_and_highlight_shadow(image_path, model, target_size=(128, 128)):

    original_img = cv2.imread(image_path)
    img_resized = cv2.resize(original_img, target_size)

    features = extract_features(img_resized)

    predictions = model.predict(features).reshape(target_size)
    predictions_normalized = cv2.normalize(predictions, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, thresholded_mask = cv2.threshold(predictions_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened_mask = cv2.morphologyEx(thresholded_mask, cv2.MORPH_OPEN, kernel)
    closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)

    thresholded_mask_resized = cv2.resize(closed_mask, (original_img.shape[1], original_img.shape[0]))

    shadow_colored = original_img.copy()
    shadow_colored[thresholded_mask_resized == 255] = [0, 0, 255]

    cv2.imwrite(os.path.join(app.config['STATIC_FOLDER'], 'shadow_thresholded_image.jpg'), thresholded_mask_resized)

    return shadow_colored, thresholded_mask_resized

def remove_shadow_from_image(image, thresholded_mask):
    dilation_size = 4
    corrected_img = image.copy()
    height, width, channels = corrected_img.shape

    print(f"Height: {height}, Width: {width}, Channels: {channels}")

    # Create a kernel for dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
    expanded_mask = cv2.dilate(thresholded_mask, kernel, iterations=1)

    # Calculate mean shadow intensity for each channel
    shadow_pixels_r = image[expanded_mask == 0, 0]
    shadow_pixels_g = image[expanded_mask == 0, 1]
    shadow_pixels_b = image[expanded_mask == 0, 2]

    mean_shadow_intensity_r = np.mean(shadow_pixels_r) / 255.0 if len(shadow_pixels_r) > 0 else 0.5
    mean_shadow_intensity_g = np.mean(shadow_pixels_g) / 255.0 if len(shadow_pixels_g) > 0 else 0.5
    mean_shadow_intensity_b = np.mean(shadow_pixels_b) / 255.0 if len(shadow_pixels_b) > 0 else 0.5

    # Dynamic gamma adjustment based on shadow brightness for each channel
    gamma_r = 0.2 + (1.0 - mean_shadow_intensity_r) * 0.8
    gamma_g = 0.1 + (1.0 - mean_shadow_intensity_g) * 0.9
    gamma_b = 0.1 + (1.0 - mean_shadow_intensity_b) * 0.9

    print(f"Gamma R: {gamma_r}, Gamma G: {gamma_g}, Gamma B: {gamma_b}")

    # Apply gamma correction separately for each channel
    for i in range(width):
        for j in range(height):
            if expanded_mask[j, i] == 0:  # Shadow region
                corrected_img[j, i, 0] = 255
                corrected_img[j, i, 1] = 255
                corrected_img[j, i, 2] = 255
            else:  # Non-shadow region
                corrected_img[j, i, 0] = np.power((corrected_img[j, i, 0] / 255), gamma_r) * 255
                corrected_img[j, i, 1] = np.power((corrected_img[j, i, 1] / 255), gamma_g) * 255
                corrected_img[j, i, 2] = np.power((corrected_img[j, i, 2] / 255), gamma_b) * 255

    # Restore background pixels as they are
    for i in range(width):
        for j in range(height):
            if expanded_mask[j, i] == 0:  
                corrected_img[j, i, 0] = image[j, i, 0]
                corrected_img[j, i, 1] = image[j, i, 1]
                corrected_img[j, i, 2] = image[j, i, 2]

    # Smooth edges using a Gaussian blur
    blurred_mask = cv2.GaussianBlur(expanded_mask.astype(np.float32), (15, 15), 10)

    for c in range(channels):
        corrected_img[..., c] = (corrected_img[..., c] * blurred_mask / 255.0 +
                                 image[..., c] * (1 - blurred_mask / 255.0)).astype(np.uint8)

    return corrected_img

if __name__ == '__main__':
    app.run(debug=True)