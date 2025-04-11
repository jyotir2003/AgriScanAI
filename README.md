# 🌿 Plant Disease Recognition System 🔍

![Plant Disease Classification](home_page.jpeg)

## Project Overview 📋
This system uses deep learning to identify 38 different types of plant diseases from images. Built with TensorFlow and Streamlit, it provides an easy-to-use interface for farmers and gardeners to diagnose plant health issues quickly and accurately. The AI-powered solution helps in early detection of plant diseases, potentially saving crops and reducing the need for pesticides.

## Features ✨
- 🔍 **Disease Recognition**: Upload images of plant leaves to detect diseases
- 🌱 **38 Disease Categories**: Covering common crops including Apple, Tomato, Potato, Corn, Grape, and more
- 📊 **High Accuracy**: ~95% accuracy on validation data
- 💻 **User-friendly Interface**: Simple web application built with Streamlit
- 🖼️ **Image Preview**: View uploaded images before prediction
- 📱 **Responsive Design**: Works on desktop and mobile devices

## Dataset 📊
The model is trained on the [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) from Kaggle which includes:
- 📷 87,000+ images (both healthy and diseased plants)
- 🏷️ 38 different disease categories
- 📑 Split into training (70,000+), validation (17,500+), and test sets
- 🌿 High-quality RGB images of various plant leaves

## Model Architecture 🧠
The system uses a Convolutional Neural Network (CNN) with:
- 🔢 **Input Layer**: Accepts RGB images (128x128x3)
- 🔍 **Feature Extraction**: 
  - 5 blocks of convolutional layers (32→512 filters)
  - Each block contains 2 Conv2D layers followed by MaxPooling
  - ReLU activation functions for non-linearity
- 🛑 **Regularization**: Dropout layers (0.25, 0.4) to prevent overfitting
- 🧮 **Classification Head**: 
  - Flatten layer to convert 2D features to 1D
  - Dense layer with 1500 neurons
  - Final layer with 38 outputs and softmax activation
- ⚙️ **Training**: 10 epochs with Adam optimizer (learning rate=0.0001)
- 📈 **Performance**: 97.8% training accuracy and 94.6% validation accuracy

## Technical Stack 🛠️
- **Deep Learning**: TensorFlow/Keras - Powers the CNN model for image classification
- **Web Interface**: Streamlit - Creates an interactive web application with minimal code
- **Image Processing**: PIL, OpenCV - Handle image loading, resizing, and preprocessing
- **Data Analysis**: NumPy, Pandas - Support data manipulation and analysis
- **Visualization**: Matplotlib, Seaborn - Generate insightful charts and plots
- **Model Evaluation**: Scikit-learn - Calculate precision, recall, F1-score metrics

## Directory Structure �

## Installation 📥

### Prerequisites
- Python 3.8+
- Pip package manager

### Setup
1. Clone this repository
```bash
git clone https://github.com/yourusername/plant-disease-recognition.git
cd plant-disease-recognition
```

2. Create and activate a virtual environment (recommended)
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages
```bash
pip install -r requirements.txt
```

4. Create a requirements.txt file with the following content:
```
tensorflow>=2.9.0
streamlit>=1.20.0
numpy>=1.20.0
pillow>=9.0.0
opencv-python>=4.6.0
matplotlib>=3.5.0
pandas>=1.4.0
scikit-learn>=1.0.0
```

## Usage 🚀

### Step 1: Prepare your environment
Make sure you've completed all installation steps and your virtual environment is activated.

### Step 2: Download the model
Ensure the trained model file `trained_plant_disease_model.keras` is in your project directory. If you're using your own model, update the path in `main.py`.

### Step 3: Run the Streamlit app
```bash
streamlit run main.py
```
This will start the web server and automatically open your default browser to the app (typically at http://localhost:8501).

### Step 4: Using the application
1. Navigate to the "Disease Recognition" page using the sidebar menu
2. Upload an image of a plant leaf using the file uploader
3. Click "Show Image" to view the uploaded image
4. Click "Predict" to analyze the image and get the disease classification result
5. Review the prediction result that appears below the buttons

## Model Performance 📈
- ✅ **Training Accuracy**: 97.8%
- 🔍 **Validation Accuracy**: 94.6%
- 📊 **F1-Score**: 0.95 (weighted average)
- 📉 **Loss Function**: Categorical Cross-Entropy
- ⚖️ **Precision/Recall**: High performance across all classes (see classification report in documentation)
- 🎯 **Confusion Matrix**: Strong diagonal indicating good class separation

## Disease Categories
The system can identify the following plant diseases:
- Apple: Apple Scab, Black Rot, Cedar Apple Rust, Healthy
- Blueberry: Healthy
- Cherry: Powdery Mildew, Healthy
- Corn: Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy
- Grape: Black Rot, Esca (Black Measles), Leaf Blight, Healthy
- Orange: Haunglongbing (Citrus Greening)
- Peach: Bacterial Spot, Healthy
- Pepper: Bacterial Spot, Healthy
- Potato: Early Blight, Late Blight, Healthy
- Raspberry: Healthy
- Soybean: Healthy
- Squash: Powdery Mildew
- Strawberry: Leaf Scorch, Healthy
- Tomato: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy

## Future Improvements 🚀
- 💊 **Treatment Recommendations**: Add disease-specific treatment advice
- 🌱 **More Plant Species**: Expand to additional crops and ornamental plants
- 🔄 **Transfer Learning**: Test newer model architectures like EfficientNet or Vision Transformer
- 📱 **Mobile App**: Create Android/iOS applications for field use
- 🌐 **API Service**: Build a REST API for integration with other agricultural systems
- 🗄️ **Database**: Add historical tracking of disease outbreaks
- 🔌 **Offline Mode**: Enable functionality without internet connection

## Troubleshooting 🔧
- **Model Loading Error**: Ensure you have the correct Keras/TensorFlow version installed
- **Image Processing Error**: Make sure uploaded images are in JPG, PNG, or JPEG format
- **Memory Issues**: Try reducing batch size if you face memory limitations
- **CUDA Errors**: For GPU acceleration, verify compatible CUDA and cuDNN versions

## Contributing 👥
Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License 📄
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements 🙏
- [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) on Kaggle
- TensorFlow and Keras community
- Streamlit community
- All contributors to the project

## Contact 📧
For questions or feedback, please contact [jyotiranjan.3.behera@gmail.com]
