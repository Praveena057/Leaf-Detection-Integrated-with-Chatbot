import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import tensorflow as tf
import numpy as np

# Load the pre-trained leaf recognition model
model = tf.keras.models.load_model('leaf_model.keras') 

# Define leaf names corresponding to the model's output classes
leaf_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy ', 'Potato___Early_blight', 
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]


def classify_image(image_path):
    try:
        # Load and preprocess the input image
        input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
        input_image_array = tf.keras.utils.img_to_array(input_image)
        input_image_exp_dim = np.expand_dims(input_image_array, axis=0)

        # Make predictions on the input image
        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0])

        # Determine the predicted leaf class and confidence score
        predicted_class_index = np.argmax(result)
        predicted_leaf = leaf_names[predicted_class_index]
        confidence_score = np.max(result) * 100

        return predicted_leaf, confidence_score
    except Exception as e:
        print(f"Error classifying image: {e}")
        return None, None

def classify_images_in_directory(directory_path):
    predictions = []
    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)
        leaf_class, confidence = classify_image(image_path)
        if leaf_class is not None and confidence is not None:
            predictions.append((image_file, leaf_class, confidence))
    
    return predictions

def select_directory():
    directory_path = filedialog.askdirectory()
    if directory_path:
        predictions = classify_images_in_directory(directory_path)
        if predictions:
            display_predictions(directory_path, predictions)
        else:
            messagebox.showinfo("Info", "No valid images found for classification.")

def display_predictions(directory_path, predictions):
    new_window = tk.Toplevel(root)
    new_window.title("Image Classifications")

    canvas = tk.Canvas(new_window)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar = ttk.Scrollbar(new_window, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Configure canvas to use scrollbar
    canvas.configure(yscrollcommand=scrollbar.set)

    # Create a frame inside the canvas for the images
    frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=frame, anchor=tk.NW)

    # Display images and classifications in the frame
    for i, (image_file, leaf_class, confidence) in enumerate(predictions):
        img = Image.open(os.path.join(directory_path, image_file))
        img = img.resize((200, 200))  # Resize for display
        photo = ImageTk.PhotoImage(img)

        label_text = f"{image_file} - {leaf_class} (Confidence: {confidence:.2f}%)"
        label = tk.Label(frame, text=label_text)
        label.grid(row=i, column=0, padx=10, pady=10, sticky=tk.W)

        img_label = tk.Label(frame, image=photo)
        img_label.image = photo  # Keep a reference to avoid garbage collection
        img_label.grid(row=i, column=1, padx=10, pady=10)

    # Update the canvas scroll region
    frame.update_idletasks()  # Ensure all widgets are updated and visible
    canvas.configure(scrollregion=canvas.bbox(tk.ALL))

# Create main application window
root = tk.Tk()
root.title("Leaf Detection and Classification")

# Create a button to select directory
select_button = tk.Button(root, text="Select Directory", command=select_directory)
select_button.pack(pady=20)

# Run the main event loop
root.mainloop()
