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

    rgb_flat = image.reshape(-1, 3)  # RGB values
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

def train_shadow_removal_model():
    # Dataset paths
    shadow_images = glob('ISTD_Dataset/train/train_A/*')
    shadow_masks = glob('ISTD_Dataset/train/train_B/*')
    shadow_free_images = glob('ISTD_Dataset/train/train_C/*')

    def load_shadow_removal_data(shadow_images, shadow_masks, shadow_free_images, target_size=(x_dim, y_dim), max_images=max_images_value):
        X, Y = [], []

        shadow_images = shadow_images[:max_images]
        shadow_masks = shadow_masks[:max_images]
        shadow_free_images = shadow_free_images[:max_images]

        for img_path, mask_path, free_path in zip(shadow_images, shadow_masks, shadow_free_images):
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            shadow_free = cv2.imread(free_path)

            img = resize_image(img, target_size)
            mask = resize_image(mask, target_size)
            shadow_free = resize_image(shadow_free, target_size)

            # Extract RGB features for shadowed regions only
            for i in range(target_size[0]):
                for j in range(target_size[1]):
                    if mask[i, j] > 0:  # Shadow region
                        pixel_features = img[i, j, :]  # Only RGB values
                        X.append(pixel_features)  # Features for the shadowed pixel
                        Y.append(shadow_free[i, j, :])  # Corresponding shadow-free RGB value

        X = np.array(X)
        Y = np.array(Y)
        return X, Y

    X, Y = load_shadow_removal_data(shadow_images, shadow_masks, shadow_free_images)

    # Train a regression model to predict shadow-free pixel values
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators=50, random_state=42)
    regressor.fit(X, Y)

    print("Shadow removal model trained.")
    return regressor

def remove_shadows(image_path, shadow_model, target_size=(x_dim, y_dim)):
    original_img = cv2.imread(image_path)
    img_resized = resize_image(original_img, target_size)

    # Prepare a new image for the shadow-free output
    shadow_free_img = img_resized.copy()

    # Iterate through each pixel and predict the shadow-free RGB values
    for i in range(target_size[0]):
        for j in range(target_size[1]):
            # Extract only the RGB features (3 values per pixel)
            pixel_features = img_resized[i, j, :]  # This is a 1D array of length 3 (RGB)
            
            # Ensure that you're passing the features as a 2D array with shape (1, 3)
            predicted_pixel = shadow_model.predict([pixel_features])[0]  # Predict for the single pixel
            
            # Assign the predicted shadow-free pixel value to the output image
            shadow_free_img[i, j, :] = predicted_pixel

    # Resize the shadow-free image to the original size of the input image
    shadow_free_img_resized = cv2.resize(shadow_free_img, (original_img.shape[1], original_img.shape[0]))

    # Save the shadow-free image
    cv2.imwrite("shadow_removed_image.jpg", shadow_free_img_resized)

    return shadow_free_img_resized



def remove_shadow():
    global shadow_removal_model
    if not filepath:
        messagebox.showerror("Error", "No file selected")
        return

    if shadow_removal_model is None:
        if messagebox.askyesno("Train Model", "The shadow removal model is not trained yet. Do you want to train it now?"):
            shadow_removal_model = train_shadow_removal_model()
        else:
            return

    shadow_free_img = remove_shadows(filepath, shadow_removal_model)
    shadow_free_img = cv2.cvtColor(shadow_free_img, cv2.COLOR_BGR2RGB)

    # Display shadow-free image
    shadow_free_img = Image.fromarray(shadow_free_img)
    shadow_free_img = ImageTk.PhotoImage(shadow_free_img)

    panel.configure(image=shadow_free_img)
    panel.image = shadow_free_img
    messagebox.showinfo("Info", "Shadow removal completed. Processed image saved as 'shadow_removed_image.jpg'.")


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
    cv2.imwrite("shadow_thresholded_image.jpg", closed_mask)
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

btn_remove = tk.Button(frame, text="Remove Shadow", command=remove_shadow)
btn_remove.grid(row=0, column=2, padx=10)

panel = tk.Label(app)
panel.pack(pady=10)

app.mainloop()
#train_model() 