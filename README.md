# MNIST Digit Recognition with CNN

This is my machine learning project where I built a Convolutional Neural Network to recognize handwritten digits (0 through 9). The model is trained on the MNIST dataset and I also made a small web app using Streamlit where you can draw a digit or upload an image and it'll tell you what number it is.


## What this project does

You draw a number on a canvas (or upload a photo of one), and the trained CNN model predicts which digit it is along with how confident it is. Pretty straightforward.

The model gets around 99% accuracy on the test set which is solid for MNIST.


## Tech stack

- **Python 3.13** — main language
- **TensorFlow / Keras** — for building and training the CNN
- **Streamlit** — for the web interface
- **streamlit-drawable-canvas** — the drawing pad component
- **Matplotlib** — for plotting training graphs and confusion matrix
- **Scikit-learn** — for evaluation metrics (precision, recall, f1, etc.)
- **Pillow** — image processing
- **NumPy** — number crunching

Everything runs on CPU, no GPU needed.


## About the dataset

I used the standard MNIST dataset that comes built into Keras. It has 70,000 grayscale images of handwritten digits, each 28x28 pixels. 60k are for training and 10k for testing. It downloads automatically the first time you run the training script.


## How the CNN works

The network has three convolutional blocks. Each block has a Conv2D layer with ReLU activation followed by MaxPooling to reduce the spatial dimensions. After that, the feature maps get flattened and passed through a dense layer (256 neurons) with dropout to prevent overfitting. The final layer has 10 neurons with softmax — one for each digit.

```
Input (28x28x1 grayscale image)
  → Conv2D(32 filters, 3x3) + ReLU → MaxPool(2x2)
  → Conv2D(64 filters, 3x3) + ReLU → MaxPool(2x2)
  → Conv2D(128 filters, 3x3) + ReLU → MaxPool(2x2)
  → Flatten
  → Dense(256) + ReLU + Dropout(0.3)
  → Dense(10) + Softmax
```

I used Adam optimizer with sparse categorical crossentropy as the loss function.


## Data preprocessing

Before feeding images into the model I did a few things:
- Normalized pixel values from 0-255 down to 0-1 (helps the model converge faster)
- Reshaped images to include a channel dimension since Conv2D expects that
- Split off 10% of the training data as a validation set

## Data augmentation

To help the model generalize better I added some random transformations during training — small rotations (about ±6 degrees), slight shifts up/down/left/right, and minor zoom. These get applied on the fly so the model sees slightly different versions of each image every epoch.

I specifically did NOT add horizontal or vertical flipping because flipping a 6 turns it into a 9 (and vice versa), which would mess up the labels.


## Training details

- Trains for up to 30 epochs but has early stopping that kicks in after at least 10 epochs
- Early stopping watches validation loss and stops if it hasn't improved for 5 epochs
- ModelCheckpoint saves the best model automatically so even if training goes on too long we keep the best weights
- Batch size is 64


## Project structure

```
B:\ML\
├── app.py               ← the Streamlit web app
├── train.py             ← run this to train the model
├── evaluate.py          ← run this to get metrics and plots
├── model.py             ← CNN architecture definition
├── preprocessing.py     ← data loading, normalization, augmentation
├── requirements.txt     ← pip dependencies
├── README.md            ← you're reading this
├── models/
│   ├── mnist_model.keras    ← saved trained model
│   └── history.npy          ← training history for plotting
├── results/
│   ├── training_plots.png       ← accuracy & loss curves
│   ├── confusion_matrix.png     ← confusion matrix heatmap
│   └── classification_report.txt
└── utils/
    └── helpers.py       ← image preprocessing, prediction helpers
```


## How to set it up and run it

Open PowerShell and do the following:

**1. Go to the project folder**
```powershell
cd B:\ML
```

**2. Create a virtual environment** (only need to do this once)
```powershell
python -m venv venv
```

**3. Activate the virtual environment**
```powershell
.\venv\Scripts\Activate
```
You should see `(venv)` at the start of your prompt after this.

**4. Install the dependencies**
```powershell
pip install -r requirements.txt
```
This takes a few minutes since TensorFlow is a big package.

**5. Train the model**
```powershell
python train.py
```
This downloads the MNIST data on first run and then trains the CNN. Takes maybe 5-15 minutes on CPU depending on your machine. The best model gets saved to `models/mnist_model.keras`.

**6. Evaluate the model**
```powershell
python evaluate.py
```
This runs the model on the test set and generates the confusion matrix, classification report, and training graphs in the `results/` folder.

**7. Launch the web app**
```powershell
streamlit run app.py
```
After a couple seconds you'll see something like:
```
  Local URL: http://localhost:8501
```
Open that link in Chrome or Edge and you're good to go. If that port is busy it'll use 8502 or 8503 instead — just check what it says in the terminal.

To stop the app, hit `Ctrl+C` in the terminal.


## What the web app looks like

There are two tabs:

**Draw Digit** — there's a canvas where you can draw with your mouse. Hit "Predict from Drawing" and it shows you the predicted digit, confidence percentage, and a bar chart of probabilities for all 10 digits. The "Clear Canvas" button wipes it so you can try again.

**Upload Image** — you can upload a PNG or JPG of a handwritten digit. It handles the resizing and grayscale conversion automatically.


## Evaluation metrics

The model is evaluated using accuracy, precision, recall, and F1-score (all weighted). There's also a full per-class classification report and a confusion matrix heatmap so you can see which digits it confuses with each other.

All metrics come from actual model predictions on the test set — nothing is hardcoded.


## Things to note

- You need to train the model before running the app (the app loads a saved model, it doesn't train from scratch)
- If you get a "module not found" error you probably forgot to activate the venv
- The model runs on CPU so no CUDA or GPU drivers needed
- Tested on Windows with Python 3.13


## What could be improved

- Could add more convolutional layers or use batch normalization for a small accuracy boost
- The web app is pretty basic — could add a history of past predictions
- Could try other datasets beyond MNIST (like EMNIST for letters)
- Mobile-friendly canvas would be nice but that's more of a frontend thing