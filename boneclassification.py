import tkinter as tk
from tkinter import filedialog, messagebox, Label, Button, Text, Scrollbar, Canvas, Toplevel
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from PIL import Image, ImageTk

# Initialize main window
main = tk.Tk()
main.title("Bone Fracture Classification System")
main.geometry("1200x800")
main.config(bg='#2C3E50')

# Global variables
global filename, X, Y, X_train, X_test, y_train, y_test, classifier
labels = ['Chest', 'Elbow', 'Finger', 'Hand', 'Head', 'Shoulder', 'Wrist']

# Mapping of categories to specific bones and their regions (simulated ROIs)
bone_info = {
    'Chest': {
        'bones': [
            {'name': 'Ribs', 'roi': (50, 50, 200, 300), 'description': 'The ribs form the rib cage, protecting the heart and lungs.'},
            {'name': 'Sternum', 'roi': (200, 100, 250, 250), 'description': 'The sternum (breastbone) is a flat bone in the center of the chest.'}
        ]
    },
    'Elbow': {
        'bones': [
            {'name': 'Humerus', 'roi': (50, 50, 150, 200), 'description': 'The humerus is the long bone in the upper arm.'},
            {'name': 'Radius', 'roi': (150, 200, 250, 300), 'description': 'The radius is one of the two long bones in the forearm, on the thumb side.'},
            {'name': 'Ulna', 'roi': (100, 200, 200, 350), 'description': 'The ulna is the other long bone in the forearm, on the pinky side.'}
        ]
    },
    'Finger': {
        'bones': [
            {'name': 'Phalanges', 'roi': (100, 100, 300, 300), 'description': 'The phalanges of the hand are the finger bones.'}
        ]
    },
    'Hand': {
        'bones': [
            {'name': 'Metacarpals', 'roi': (100, 50, 300, 150), 'description': 'The metacarpals are the bones in the palm of the hand.'},
            {'name': 'Carpals', 'roi': (50, 150, 200, 250), 'description': 'The carpals are the wrist bones, connecting the hand to the forearm.'}
        ]
    },
    'Head': {
        'bones': [
            {'name': 'Skull', 'roi': (50, 50, 300, 300), 'description': 'The skull protects the brain and forms the structure of the head.'},
            {'name': 'Mandible', 'roi': (100, 300, 250, 350), 'description': 'The mandible is the lower jawbone, the largest bone in the human face.'}
        ]
    },
    'Shoulder': {
        'bones': [
            {'name': 'Clavicle', 'roi': (50, 50, 350, 100), 'description': 'The clavicle (collarbone) connects the arm to the body.'},
            {'name': 'Scapula', 'roi': (50, 100, 200, 300), 'description': 'The scapula (shoulder blade) connects the humerus with the clavicle.'}
        ]
    },
    'Wrist': {
        'bones': [
            {'name': 'Carpals', 'roi': (100, 100, 300, 300), 'description': 'The carpals are the wrist bones, connecting the hand to the forearm.'},
            {'name': 'Distal Radius', 'roi': (50, 50, 150, 200), 'description': 'The distal radius is the end of the radius bone near the wrist.'},
            {'name': 'Distal Ulna', 'roi': (150, 50, 250, 200), 'description': 'The distal ulna is the end of the ulna bone near the wrist.'}
        ]
    }
}

# Color scheme
BG_COLOR = '#2C3E50'
FG_COLOR = '#ECF0F1'
ACCENT_COLOR = '#3498DB'
SECONDARY_COLOR = '#E74C3C'
CARD_COLOR = '#34495E'

def preprocess_dataset(dataset_path):
    """
    Preprocess the dataset by loading images, extracting features, and saving them to X.txt.npy and Y.txt.npy.
    Returns True if successful, False otherwise.
    """
    global labels, X, Y
    X_data = []
    Y_data = []
    
    if not os.path.exists('model'):
        os.makedirs('model')
    
    text.insert(tk.END, "Starting dataset preprocessing...\n")
    total_images = 0
    skipped_images = 0
    
    for label_idx, label in enumerate(labels):
        folder_path = os.path.join(dataset_path, label)
        if not os.path.exists(folder_path):
            text.insert(tk.END, f"Warning: Folder '{label}' not found in dataset!\n")
            continue
        
        text.insert(tk.END, f"Processing folder: {label}\n")
        folder_images = 0
        
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                text.insert(tk.END, f"Skipping non-image file: {img_name}\n")
                skipped_images += 1
                continue
            
            try:
                img = cv2.imread(img_path)
                if img is None:
                    text.insert(tk.END, f"Failed to load image: {img_name}\n")
                    skipped_images += 1
                    continue
                
                img = cv2.resize(img, (32, 32))
                img = img.astype('float32') / 255.0
                img_flat = img.ravel()
                
                X_data.append(img_flat)
                Y_data.append(label_idx)
                folder_images += 1
                total_images += 1
            except Exception as e:
                text.insert(tk.END, f"Error processing {img_name}: {str(e)}\n")
                skipped_images += 1
                continue
        
        text.insert(tk.END, f"Processed {folder_images} images in {label} folder\n")
    
    if total_images == 0:
        text.insert(tk.END, "Error: No valid images found in the dataset!\n")
        return False
    
    X = np.array(X_data)
    Y = np.array(Y_data)
    
    np.save('model/X.txt.npy', X)
    np.save('model/Y.txt.npy', Y)
    
    text.insert(tk.END, f"Preprocessing complete!\n")
    text.insert(tk.END, f"Total images processed: {total_images}\n")
    text.insert(tk.END, f"Images skipped (invalid or failed to load): {skipped_images}\n")
    text.insert(tk.END, f"Features per image: {X.shape[1] if X.size > 0 else 0}\n")
    
    if X.size > 0:
        test = X[3] if len(X) > 3 else X[0]
        test = test.reshape(32, 32, 3)
        test = cv2.resize(test, (250, 250))
        cv2.imshow("Sample Image from Dataset", test)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return True

def uploadDataset():
    global filename
    try:
        filename = filedialog.askdirectory(
            initialdir=".",
            title="Select Dataset Directory"
        )
        if not filename:
            messagebox.showwarning("Warning", "No directory selected!")
            status_label.config(text="No directory selected")
            return
        
        expected_folders = set(labels)
        found_folders = set(os.listdir(filename)) if os.path.isdir(filename) else set()
        
        if not expected_folders.issubset(found_folders):
            messagebox.showerror("Error", "Dataset directory must contain folders: " + ", ".join(labels))
            status_label.config(text="Invalid dataset structure")
            return
        
        text.delete('1.0', tk.END)
        text.insert(tk.END, f"Dataset loaded from: {filename}\n")
        text.insert(tk.END, f"Found folders: {', '.join(found_folders)}\n")
        
        success = preprocess_dataset(filename)
        if success:
            pathlabel.config(text=filename)
            status_label.config(text="Dataset preprocessed and loaded")
        else:
            messagebox.showerror("Error", "Failed to preprocess dataset. Check the console for details.")
            status_label.config(text="Preprocessing failed")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
        status_label.config(text="Error loading dataset")

def trainTestGenerator():
    text.delete('1.0', tk.END)
    global X, Y, X_train, X_test, y_train, y_test
    if 'X' not in globals() or 'Y' not in globals():
        messagebox.showerror("Error", "Please upload and preprocess the dataset first!")
        return
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    text.insert(tk.END, "Total images in dataset: " + str(X.shape[0]) + "\n")
    text.insert(tk.END, "Training images: " + str(X_train.shape[0]) + "\n")
    text.insert(tk.END, "Testing images: " + str(X_test.shape[0]) + "\n")
    status_label.config(text="Train/Test split completed")

def randomForest():
    text.delete('1.0', tk.END)
    global X_train, X_test, y_train, y_test, classifier
    if 'X_train' not in globals():
        messagebox.showerror("Error", "Please generate train/test data first!")
        return
    try:
        if os.path.exists('model/model.txt'):
            with open('model/model.txt', 'rb') as file:
                classifier = pickle.load(file)
            text.insert(tk.END, "Loaded pre-trained Random Forest model\n")
        else:
            classifier = RandomForestClassifier(n_estimators=200, random_state=0)
            classifier.fit(X_train, y_train)
            with open('model/model.txt', 'wb') as file:
                pickle.dump(classifier, file)
            text.insert(tk.END, "Trained and saved new Random Forest model\n")
        
        predict = classifier.predict(X_test)
        random_acc = accuracy_score(y_test, predict) * 100
        text.insert(tk.END, f"Random Forest Accuracy: {random_acc:.2f}%\n\n")
        text.insert(tk.END, "Classification Report:\n\n")
        text.insert(tk.END, classification_report(y_test, predict, target_names=labels) + "\n")
        status_label.config(text="Random Forest model trained")
    except Exception as e:
        messagebox.showerror("Error", f"Error training model: {str(e)}")

class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        x, y = self.widget.winfo_pointerxy()
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x+10}+{y+10}")
        label = tk.Label(self.tooltip_window, text=self.text, background="yellow", relief="solid", borderwidth=1)
        label.pack()

    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

def predict():
    global classifier
    if 'classifier' not in globals():
        messagebox.showerror("Error", "Please train the model first!")
        return
    filename = filedialog.askopenfilename(initialdir="testImages", title="Select Test Image",
                                        filetypes=[("Image files", "*.png *.jpg *.jpeg")])
    if filename:
        try:
            img = cv2.imread(filename)
            if img is None:
                raise ValueError("Failed to load image!")
            img_resized = cv2.resize(img, (32, 32))
            test = np.array(img_resized, dtype='float32') / 255
            test = test.ravel()
            test_data = np.array([test])
            prediction = classifier.predict(test_data)[0]
            
            category = labels[prediction]
            bones = bone_info[category]['bones']
            
            text.insert(tk.END, f"Predicted Category: {category}\n")
            text.insert(tk.END, "Bones in this region:\n")
            for bone in bones:
                text.insert(tk.END, f"- {bone['name']}: {bone['description']}\n")
            text.insert(tk.END, "\n")
            
            # Create a new window for interactive display
            interactive_window = Toplevel(main)
            interactive_window.title(f"Classified as: {category}")
            interactive_window.geometry("600x600")
            
            # Load and display the image
            img_display = cv2.resize(img, (400, 400))
            img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_display)
            img_tk = ImageTk.PhotoImage(img_pil)
            
            canvas = Canvas(interactive_window, width=400, height=400)
            canvas.pack(pady=10)
            canvas.create_image(0, 0, anchor="nw", image=img_tk)
            
            # Draw ROIs for each bone in the category
            for bone in bones:
                x1, y1, x2, y2 = bone['roi']
                region = canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)
                # Add tooltip for the bone region
                canvas.tag_bind(region, "<Enter>", lambda e, name=bone['name']: show_tooltip(e, name))
                canvas.tag_bind(region, "<Leave>", lambda e: hide_tooltip())
            
            # Keep a reference to the image to prevent garbage collection
            canvas.img_tk = img_tk
            
            # Add description below the image
            desc_text = f"Category: {category}\nBones in this region:\n"
            for bone in bones:
                desc_text += f"- {bone['name']}: {bone['description']}\n"
            desc_label = Label(interactive_window, text=desc_text, wraplength=500, justify="left")
            desc_label.pack(pady=10)
            
            status_label.config(text="Prediction completed")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")

# Global variables for tooltip
tooltip_window = None

def show_tooltip(event, text):
    global tooltip_window
    x, y = event.widget.winfo_pointerxy()
    tooltip_window = tk.Toplevel(event.widget)
    tooltip_window.wm_overrideredirect(True)
    tooltip_window.wm_geometry(f"+{x+10}+{y+10}")
    label = tk.Label(tooltip_window, text=text, background="yellow", relief="solid", borderwidth=1)
    label.pack()

def hide_tooltip():
    global tooltip_window
    if tooltip_window:
        tooltip_window.destroy()
        tooltip_window = None

# GUI Setup
title_frame = tk.Frame(main, bg=ACCENT_COLOR, bd=0)
title_frame.pack(fill='x', pady=(0, 20))
title = Label(title_frame, text='Bone Fracture Classification System', bg=ACCENT_COLOR, fg=FG_COLOR,
              font=('Helvetica', 24, 'bold'), pady=20)
title.pack()

content_frame = tk.Frame(main, bg=BG_COLOR)
content_frame.pack(fill='both', expand=True, padx=20, pady=10)

control_frame = tk.Frame(content_frame, bg=CARD_COLOR, relief='raised', bd=2)
control_frame.pack(side='left', fill='y', padx=(0, 10), pady=10)

btn_config = {'bg': ACCENT_COLOR, 'fg': FG_COLOR, 'font': ('Helvetica', 12, 'bold'),
              'activebackground': '#2980B9', 'width': 25, 'height': 2, 'bd': 0, 'cursor': 'hand2'}

btn1 = Button(control_frame, text="1. Upload & Preprocess Dataset", command=uploadDataset, **btn_config)
btn1.pack(pady=10, padx=20)
pathlabel = Label(control_frame, text="No dataset selected", bg=CARD_COLOR, fg=FG_COLOR,
                 font=('Helvetica', 10), wraplength=200)
pathlabel.pack(pady=5)

btn3 = Button(control_frame, text="2. Generate Train/Test Data", command=trainTestGenerator, **btn_config)
btn3.pack(pady=10, padx=20)
btn4 = Button(control_frame, text="3. Train RF Model", command=randomForest, **btn_config)
btn4.pack(pady=10, padx=20)
btn5 = Button(control_frame, text="4. Classify Test Image", command=predict, **btn_config)
btn5.pack(pady=10, padx=20)

output_frame = tk.Frame(content_frame, bg=CARD_COLOR, relief='raised', bd=2)
output_frame.pack(side='right', fill='both', expand=True, pady=10)
output_title = Label(output_frame, text="Output Console", bg=CARD_COLOR, fg=FG_COLOR,
                    font=('Helvetica', 14, 'bold'), pady=10)
output_title.pack(fill='x')

text = Text(output_frame, height=30, width=80, bg=FG_COLOR, fg=BG_COLOR, font=('Helvetica', 11), bd=0)
text.pack(side='left', fill='both', expand=True, padx=10, pady=(0, 10))
scroll = Scrollbar(output_frame, command=text.yview)
scroll.pack(side='right', fill='y', pady=(0, 10))
text.configure(yscrollcommand=scroll.set)

status_frame = tk.Frame(main, bg=ACCENT_COLOR)
status_frame.pack(fill='x', side='bottom')
status_label = Label(status_frame, text="Ready", bg=ACCENT_COLOR, fg=FG_COLOR,
                    font=('Helvetica', 10), pady=5)
status_label.pack(side='left', padx=10)

main.mainloop()