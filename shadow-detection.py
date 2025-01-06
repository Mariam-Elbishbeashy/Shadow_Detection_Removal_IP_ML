import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

from PIL import Image, ImageTk
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from glob import glob
import joblib
import os

sizing = 1
x_dim = int(640/sizing)
y_dim = int(480/sizing)
max_images_value = 150
def extract_features(image):
    h, w, _ = image.shape

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Local Binary Pattern (LBP)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform').flatten()

    #rgb_flat = image.reshape(-1, 3)  # RGB values
    gray_flat = gray.flatten()  # Grayscale intensity

    # Combine RGB, Grayscale, and LBP features
    #features = np.hstack([rgb_flat, gray_flat[:, None], lbp[:, None]])
    features = np.hstack([ gray_flat[:, None], lbp[:, None]])

    return features

def resize_image(img, target_size=(x_dim, y_dim)):
    return cv2.resize(img, target_size)

def train_model():

    image_paths = glob('ISTD_Dataset/train/train_A/*')
    mask_paths = glob('ISTD_Dataset/train/train_B/*')

    def load_dataset(image_paths, mask_paths, target_size=(x_dim, y_dim), max_images=max_images_value):
        X, Y = [], []
        image_paths = image_paths[:max_images]
        mask_paths = mask_paths[:max_images]

        for img_path, mask_path in zip(image_paths, mask_paths):
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            img = resize_image(img, target_size)
            mask = resize_image(mask, target_size)

            features = extract_features(img)
            labels = mask.flatten()

            X.append(features)
            Y.append(labels)

        X = np.vstack(X)
        Y = np.hstack(Y)
        return X, Y

    X, Y = load_dataset(image_paths, mask_paths)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=60, random_state=42,verbose=1)
    clf.fit(X_train, Y_train)

    Y_pred = clf.predict(X_test)
    print("Classification Report:")
    print(classification_report(Y_test, Y_pred, zero_division=0))
    print("Accuracy:", accuracy_score(Y_test, Y_pred))

    joblib.dump(clf, 'shadow_detection_model.pkl')
    print("Model saved as 'shadow_detection_model.pkl'")

    return clf


def remove_shadow(image_path, thresholded_mask_resized):
    dilation_size=4
    original_img = cv2.imread(image_path)

    corrected_img = original_img.copy()
    # Get the dimensions of corrected_img
    height, width, channels = corrected_img.shape

    # Print the dimensions
    print(f"Height: {height}, Width: {width}, Channels: {channels}")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
    expanded_mask = cv2.dilate(thresholded_mask_resized, kernel, iterations=1)


    shadow_pixels_r = original_img[expanded_mask[..., 0] == 0, 0]
    shadow_pixels_g = original_img[expanded_mask[..., 0] == 0, 1]
    shadow_pixels_b = original_img[expanded_mask[..., 0] == 0, 2]

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
            if expanded_mask[j, i, 0] == 0: 
                corrected_img[j, i, 0] = 255
                corrected_img[j, i, 1] = 255
                corrected_img[j, i, 2] = 255
            else:
                corrected_img[j, i, 0] = np.power((corrected_img[j, i, 0] / 255), gamma_r) * 255
                corrected_img[j, i, 1] = np.power((corrected_img[j, i, 1] / 255), gamma_g) * 255
                corrected_img[j, i, 2] = np.power((corrected_img[j, i, 2] / 255), gamma_b) * 255

    # Restore background pixels as they are
    for i in range(width):
        for j in range(height):
            if expanded_mask[j, i, 0] == 0: 
                corrected_img[j, i, 0] = original_img[j, i, 0]
                corrected_img[j, i, 1] = original_img[j, i, 1]
                corrected_img[j, i, 2] = original_img[j, i, 2]

     # Smooth edges using a Gaussian blur
    mask = expanded_mask[..., 0]
    blurred_mask = cv2.GaussianBlur(mask.astype(np.float32), (15, 15), 10)

    for c in range(channels):
        corrected_img[..., c] = (corrected_img[..., c] * blurred_mask / 255.0 +
                                 original_img[..., c] * (1 - blurred_mask / 255.0)).astype(np.uint8)


    return corrected_img


def remove_shadow_callback():
    global filepath

    if not filepath:
        messagebox.showerror("Error", "No file selected")
        return

    # Generate a shadow mask directly using Otsu's thresholding (as no model is used)
    #original_img = cv2.imread(filepath)
    thresholded_mask_resized = cv2.imread('shadow_thresholded_image.jpg')

    #gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding to generate a shadow mask
    #_, shadow_mask = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply the shadow removal function
    shadow_removed_img = remove_shadow(filepath, thresholded_mask_resized)

    # Save and display the shadow-removed image
    output_path = "shadow_removed_image.jpg"
    cv2.imwrite(output_path, shadow_removed_img)
    shadow_removed_img = cv2.cvtColor(shadow_removed_img, cv2.COLOR_BGR2RGB)
    shadow_removed_img = Image.fromarray(shadow_removed_img)
    shadow_removed_img = ImageTk.PhotoImage(shadow_removed_img)

    panel.configure(image=shadow_removed_img)
    panel.image = shadow_removed_img
    messagebox.showinfo("Info", f"Shadow removal completed. Image saved as '{output_path}'.")


    

def detect_and_highlight_shadow(image_path, model, target_size=(128, 128)):
    original_img = cv2.imread(image_path)
    img_resized = cv2.resize(original_img, target_size)

    features = extract_features(img_resized)
    predictions = model.predict(features).reshape(target_size)

    # Normalize predictions to the range [0, 255] if necessary
    predictions_normalized = cv2.normalize(predictions, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply Otsu's thresholding to find a dynamic threshold
    _, thresholded_mask = cv2.threshold(predictions_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


        # Create structuring element for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Morphological operations
    # Step 1: Remove noise (Opening)
    opened_mask = cv2.morphologyEx(thresholded_mask, cv2.MORPH_OPEN, kernel)

    # Step 2: Smooth edges and fill gaps (Closing)
    closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)

    # Resize the mask to the original image size
    thresholded_mask_resized = cv2.resize(closed_mask, (original_img.shape[1], original_img.shape[0]))

    # Highlight the shadows in the original image
    shadow_colored = original_img.copy()
    shadow_colored[thresholded_mask_resized == 255] = [0, 0, 255]  # Red color for shadows

    # Save both images
    cv2.imwrite("shadow_thresholded_image.jpg", thresholded_mask_resized)
    cv2.imwrite("highlighted_shadow_image.jpg", shadow_colored)

    return shadow_colored, thresholded_mask_resized

# Initialize model
model = None
shadow_removal_model = None

# GUI Implementation
def open_file():
    global filepath
    filepath = filedialog.askopenfilename(filetypes=[("Image Files", ".png;.jpg;*.jpeg")])
    if filepath:
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        panel.configure(image=img)
        panel.image = img


def detect_shadow():
    global model
    if not filepath:
        messagebox.showerror("Error", "No file selected")
        return

    if model is None:
        # Check if the saved model exists
        if os.path.exists('shadow_detection_model.pkl'):
            model = joblib.load('shadow_detection_model.pkl')
            print("Loaded saved model.")
        else:
            if messagebox.askyesno("Train Model", "The model is not trained yet. Do you want to train it now?"):
                model = train_model()
            else:
                return

    shadow_img, _ = detect_and_highlight_shadow(filepath, model)
    shadow_img = cv2.cvtColor(shadow_img, cv2.COLOR_BGR2RGB)
    
    # Save the processed image for verification
    cv2.imwrite("processed_shadow_image.jpg", cv2.cvtColor(shadow_img, cv2.COLOR_RGB2BGR))
    
    shadow_img = Image.fromarray(shadow_img)
    shadow_img = ImageTk.PhotoImage(shadow_img)

    panel.configure(image=shadow_img)
    panel.image = shadow_img
    messagebox.showinfo("Info", "Shadow detection completed. Processed image saved as 'processed_shadow_image.jpg'.")

app = tk.Tk()
app.title("Shadow Detection GUI")

filepath = None

# GUI Layout
frame = tk.Frame(app)
frame.pack(pady=10)

btn_select = tk.Button(frame, text="Select Image", command=open_file)
btn_select.grid(row=0, column=0, padx=10)

btn_detect = tk.Button(frame, text="Detect Shadow", command=detect_shadow)
btn_detect.grid(row=0, column=1, padx=10)

btn_remove = tk.Button(frame, text="Remove Shadow", command=remove_shadow_callback)
btn_remove.grid(row=0, column=2, padx=10)

panel = tk.Label(app)
panel.pack(pady=10)

app.mainloop()
#train_model()