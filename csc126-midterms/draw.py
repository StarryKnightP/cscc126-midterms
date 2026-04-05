# digit_app.py
# Tkinter application for real-time handwritten digit recognition.
#
# Features:
#   - Drawing canvas (mouse input)
#   - Predict button — preprocesses drawing and runs the selected model
#   - Clear button  — resets the canvas
#   - Model toggle  — switch between ANN and CNN at any time
#   - Displays predicted digit and confidence score
#
# Requirements:
#   - ann_mnist_model.keras and cnn_mnist_model.keras in the same folder
#   - pip install tensorflow pillow numpy
#
# Preprocessing matches ANN_convert.py / CNN_convert.py exactly:
#   grayscale → resize 28×28 → invert → binarize (threshold 128) → normalize
#   then reshape to (1, 784) for ANN or (1, 28, 28, 1) for CNN

import tkinter as tk
from tkinter import font as tkfont
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import keras

# =============================================================================
# CONFIG
# =============================================================================

CANVAS_SIZE   = 400        # drawing canvas width and height in pixels
ANN_MODEL_PATH = "ANN_model.h5"
CNN_MODEL_PATH = "CNN_model.h5"

# =============================================================================
# LOAD MODELS
# =============================================================================

print("Loading models...")
try:
    ann_model = keras.models.load_model(ANN_MODEL_PATH)
    print(f"  ANN loaded: '{ANN_MODEL_PATH}'")
except Exception as e:
    ann_model = None
    print(f"  WARNING: Could not load ANN model — {e}")

try:
    cnn_model = keras.models.load_model(CNN_MODEL_PATH)
    print(f"  CNN loaded: '{CNN_MODEL_PATH}'")
except Exception as e:
    cnn_model = None
    print(f"  WARNING: Could not load CNN model — {e}")

if ann_model is None and cnn_model is None:
    print("ERROR: No models could be loaded. Run ANN_train_pred.py and CNN_train_pred.py first.")
    exit()

print("Models ready.\n")

# =============================================================================
# PREPROCESSING
# Matches ANN_convert.py / CNN_convert.py pipeline exactly.
# Input : PIL Image (the canvas contents, RGBA)
# Output: numpy array shaped for the selected model
# =============================================================================

def preprocess(pil_image, model_type="CNN"):
    """
    Converts the canvas PIL image into a model-ready numpy array.

    Steps:
      1. Convert to grayscale
      2. Auto-crop to the bounding box of drawn pixels + padding
         This centers the digit the same way MNIST images are centered,
         which is the biggest source of prediction error when skipped.
      3. Resize to 28×28
      4. Invert  (canvas is white-on-black; MNIST is black-on-white)
      5. Binarize with threshold 128
      6. Normalize to [0.0, 1.0]
      7. Reshape for ANN (1, 784) or CNN (1, 28, 28, 1)
    """
    img = pil_image.convert('L')                        # step 1: grayscale

    # step 2: auto-crop + center (key fix for drawing app accuracy)
    # Find the bounding box of non-black pixels (the drawn digit)
    bbox = img.getbbox()
    if bbox:
        # Add 20px padding around the digit before resizing so it doesn't
        # touch the edges — matches how MNIST digits are framed
        pad = 20
        left  = max(bbox[0] - pad, 0)
        upper = max(bbox[1] - pad, 0)
        right  = min(bbox[2] + pad, img.width)
        lower  = min(bbox[3] + pad, img.height)
        img = img.crop((left, upper, right, lower))

        # Paste onto a square black canvas to preserve aspect ratio
        # Without this, a tall narrow digit (like 1) gets stretched wide
        side = max(img.width, img.height)
        square = Image.new('L', (side, side), 0)
        x_off = (side - img.width)  // 2
        y_off = (side - img.height) // 2
        square.paste(img, (x_off, y_off))
        img = square

    img = img.resize((28, 28), Image.LANCZOS)           # step 3: resize
    img = ImageOps.invert(img)                          # step 4: invert

    arr = np.array(img, dtype=np.float32)
    arr = np.where(arr > 128, 255.0, 0.0)              # step 5: binarize
    arr = arr / 255.0                                   # step 6: normalize

    if model_type == "ANN":
        arr = arr.reshape(1, 784)                       # step 7a: flat for ANN
    else:
        arr = arr.reshape(1, 28, 28, 1)                 # step 7b: 4-D for CNN

    return arr

# =============================================================================
# APPLICATION
# =============================================================================

class DigitApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognizer")
        self.root.resizable(False, False)

        # Track which model is active ("ANN" or "CNN")
        self.active_model = tk.StringVar(value="CNN")

        # PIL image used as the off-screen drawing buffer
        self.pil_image  = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "black")
        self.pil_draw   = ImageDraw.Draw(self.pil_image)

        self._build_ui()

    # ── UI LAYOUT ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        FONT_LABEL  = tkfont.Font(family="Helvetica", size=11)
        FONT_RESULT = tkfont.Font(family="Helvetica", size=48, weight="bold")
        FONT_CONF   = tkfont.Font(family="Helvetica", size=11)
        FONT_BTN    = tkfont.Font(family="Helvetica", size=12, weight="bold")

        # ── Left panel: canvas ─────────────────────────────────────────────
        left = tk.Frame(self.root, bg="#1e1e1e")
        left.grid(row=0, column=0, padx=16, pady=16)

        tk.Label(left, text="Draw a digit below",
                 bg="#1e1e1e", fg="#aaaaaa",
                 font=FONT_LABEL).pack(anchor="w", pady=(0, 6))

        self.canvas = tk.Canvas(
            left,
            width=CANVAS_SIZE, height=CANVAS_SIZE,
            bg="black", cursor="crosshair",
            highlightthickness=2, highlightbackground="#444444"
        )
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>",       self._on_draw)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        # ── Right panel: controls + result ────────────────────────────────
        right = tk.Frame(self.root, bg="#1e1e1e")
        right.grid(row=0, column=1, padx=(0, 16), pady=16, sticky="n")

        # Model selector
        tk.Label(right, text="Active model",
                 bg="#1e1e1e", fg="#aaaaaa",
                 font=FONT_LABEL).pack(anchor="w", pady=(0, 4))

        model_frame = tk.Frame(right, bg="#1e1e1e")
        model_frame.pack(fill="x", pady=(0, 16))

        for name in ("ANN", "CNN"):
            state = "normal" if (name == "ANN" and ann_model) or \
                                (name == "CNN" and cnn_model) else "disabled"
            tk.Radiobutton(
                model_frame, text=name,
                variable=self.active_model, value=name,
                bg="#1e1e1e", fg="white",
                selectcolor="#333333",
                activebackground="#1e1e1e", activeforeground="white",
                font=FONT_LABEL, state=state
            ).pack(side="left", padx=(0, 12))

        # Predict button
        tk.Button(
            right, text="Predict",
            command=self._predict,
            bg="#4a90d9", fg="white",
            activebackground="#357abd", activeforeground="white",
            font=FONT_BTN, relief="flat",
            width=14, pady=10
        ).pack(fill="x", pady=(0, 8))

        # Clear button
        tk.Button(
            right, text="Clear",
            command=self._clear,
            bg="#444444", fg="white",
            activebackground="#555555", activeforeground="white",
            font=FONT_BTN, relief="flat",
            width=14, pady=10
        ).pack(fill="x", pady=(0, 24))

        # Divider
        tk.Frame(right, bg="#444444", height=1).pack(fill="x", pady=(0, 20))

        # Predicted digit display
        tk.Label(right, text="Prediction",
                 bg="#1e1e1e", fg="#aaaaaa",
                 font=FONT_LABEL).pack(anchor="w")

        self.result_label = tk.Label(
            right, text="—",
            bg="#1e1e1e", fg="white",
            font=FONT_RESULT
        )
        self.result_label.pack(pady=(4, 0))

        # Confidence score
        self.conf_label = tk.Label(
            right, text="",
            bg="#1e1e1e", fg="#888888",
            font=FONT_CONF
        )
        self.conf_label.pack(pady=(0, 16))

        # Which model made the prediction
        self.model_used_label = tk.Label(
            right, text="",
            bg="#1e1e1e", fg="#555555",
            font=FONT_CONF
        )
        self.model_used_label.pack()

        # Root background
        self.root.configure(bg="#1e1e1e")

    # ── DRAWING ───────────────────────────────────────────────────────────────

    def _on_draw(self, event):
        """Draw on both the Tkinter canvas and the PIL off-screen buffer."""
        r = 14    # brush radius — thicker strokes match MNIST better after 28×28 resize
                  # at 400px canvas: r=14 → ~1.9px at 28×28, close to MNIST stroke width
                  # reduce to r=8 or r=10 if strokes feel too thick visually

        # Draw on Tkinter canvas (visible)
        self.canvas.create_oval(
            event.x - r, event.y - r,
            event.x + r, event.y + r,
            fill="white", outline="white"
        )

        # Draw on PIL buffer (used for preprocessing)
        self.pil_draw.ellipse(
            [event.x - r, event.y - r,
             event.x + r, event.y + r],
            fill="white"
        )

    def _on_release(self, event):
        """Called when the mouse button is released after drawing."""
        pass  # Auto-predict disabled — click the Predict button manually.
        # self._predict()  ← uncomment this line to re-enable auto-predict

    # ── CLEAR ─────────────────────────────────────────────────────────────────

    def _clear(self):
        """Reset the canvas and PIL buffer to black, clear result labels."""
        self.canvas.delete("all")
        self.pil_draw.rectangle(
            [0, 0, CANVAS_SIZE, CANVAS_SIZE], fill="black")

        self.result_label.config(text="—", fg="white")
        self.conf_label.config(text="")
        self.model_used_label.config(text="")

    # ── PREDICT ───────────────────────────────────────────────────────────────

    def _predict(self):
        """Preprocess the canvas contents and run the active model."""
        model_type = self.active_model.get()
        model      = ann_model if model_type == "ANN" else cnn_model

        if model is None:
            self.result_label.config(text="N/A", fg="#ff4444")
            self.conf_label.config(text=f"{model_type} model not loaded")
            return

        # Preprocess exactly as ANN_convert / CNN_convert does
        arr         = preprocess(self.pil_image, model_type)
        predictions = model.predict(arr, verbose=0)   # verbose=0 silences output
        digit       = int(np.argmax(predictions))
        confidence  = float(np.max(predictions)) * 100

        # Update display
        self.result_label.config(text=str(digit), fg="white")
        self.conf_label.config(
            text=f"{confidence:.1f}% confidence",
            fg="#4a90d9" if confidence >= 70 else "#ffaa00"
        )
        self.model_used_label.config(
            text=f"via {model_type}",
            fg="#555555"
        )

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app  = DigitApp(root)
    root.mainloop()