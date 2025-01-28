# Shadow Detection and Removal Using Image Processing and Machine Learning

This repository provides a Python-based application for detecting and removing shadows from images using a combination of image processing and machine learning techniques. It includes a GUI built with `Tkinter` for ease of use.

---

## Features

- **Shadow Detection:** Highlights shadows in the input image using a trained Random Forest classifier.
- **Shadow Removal:** Removes detected shadows and corrects the image for better visual quality.
- **GUI:** A user-friendly interface to load images, detect shadows, and remove shadows.

---

## Requirements

To run this project, you'll need the following:

- Python 3.7+
- Required Python libraries (listed in `requirements.txt`):
  - `opencv-python`
  - `scikit-image`
  - `scikit-learn`
  - `numpy`
  - `Pillow`
  - `joblib`
  - `tkinter` (comes pre-installed with Python)

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Mariam-Elbishbeashy/Shadow_Detection_Removal_IP_ML.git
   cd Shadow_Detection_Removal_IP_ML

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt

---

## Usage

### Training the Model (Optional)

If you want to retrain the shadow detection model, follow these steps:

1. **Download the ISTD Dataset** and place it in the root folder of the project.

2. **Uncomment the `train_model()` function** in `shadow-detection.py`. You can find this function in the end of the script, and it needs to be uncommented to train the model.

3. **Run the training script** to train the Random Forest model:
    ```bash
    python shadow-detection.py
    ```
    This will train a model using the dataset and save it as `shadow_detection_model.pkl`.

### Running the GUI

  To use the GUI application, **comment the `train_model()` function** in `shadow-detection.py`, then run the following command:
  
  ```bash
  python shadow-detection.py
```

---
### In the GUI, you can:

- Select an Image: Load the image you want to process.
- Detect Shadow: Highlight the detected shadows in the image.
- Remove Shadow: Remove shadows from the image and save the shadow-free version.

### File Structure
```bash
├── shadow-detection.py          # Main script with GUI and processing logic
├── ISTD_Dataset/                # Folder for the training dataset (ignored by git)
├── highlighted_shadow_image.jpg # Example output: shadow detection
├── processed_shadow_image.jpg   # Example output: processed image
├── shadow_removed_image.jpg     # Example output: shadow removed
├── shadow_thresholded_image.jpg # Example output: shadow mask
├── shadow_detection_model.pkl   # Trained Random Forest model (ignored by Git)
├── .gitignore                   # Git ignore rules
├── README.md                    # Project documentation
```
---

## Dataset:
- The dataset used for training and evaluation is the ISTD Dataset.
- Ensure the dataset is downloaded and placed correctly before running the training script.

## GUI Limitations:
- Currently, the GUI supports single-image processing only.




