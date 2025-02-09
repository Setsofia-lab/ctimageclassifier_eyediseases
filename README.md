# Eye Disease Classification using CNN models

I've developed this project to use three different CNN models to help identify different eye diseases from medical images. Think of it as a smart assistant that can look at pictures of eyes and help detect various conditions like cataracts or glaucoma.

## What Does This Project Do?

My AI system can look at eye images and identify four different conditions:
- Cataracts (clouding of the eye lens)
- Glaucoma (damage to the optic nerve)
- Diabetic Retinopathy (diabetes-related eye condition)
- Healthy Eyes (normal condition)

## How Accurate Is It?

I tested three different AI models, and here's how well they performed:

**Before Fine-tuning:**
- ResNet50: 89.7% accuracy
- VGG16: 85.9% accuracy
- MobileNetV2: 84.4% accuracy

**After Fine-tuning:**
- VGG16: 88.9% accuracy
- ResNet50: 85.9% accuracy
- MobileNetV2: 80.9% accuracy

## Getting Started

### What You'll Need
- Python 3.8 or newer
- Some key Python libraries (I'll help you install these)

### Setting Up the Project

1. **Get the Code**
   ```bash
   git clone https://github.com/Setsofia-lab/ctimageclassifier_eyediseases.git
   cd EyeDisease_Classifier_DeepLearning
   ```

2. **Set Up Your Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Required Software**
   ```bash
   pip install -r requirements.txt
   ```

## How to Use It

1. **Organize Your Images**
   Put your eye images in folders like this:
   ```
   dataset/
   ├── cataract/
   ├── glaucoma/
   ├── diabetic_retinopathy/
   └── normal/
   ```

2. **Prepare the Data**
   ```bash
   python load_dataset.py
   ```

3. **Run the AI**
   ```bash
   python model.py
   ```

## Project Organization

I've organized the project into these main folders:
- `dataset/`: Where you put your eye images
- `output_directory/`: Where the processed images go
- `scripts/`: The AI training programs
- `models/`: Where trained AI models are saved
- `notebooks/`: Interactive examples and demonstrations

## How It Works

My system uses transfer learning, which means I'm building on top of AI models that have already learned to recognize images. I've adapted these models specifically for eye disease detection by:
- Using pre-trained models that already know how to analyze images
- Teaching them to focus on specific eye disease characteristics
- Adding special layers to make them better at classifying eye conditions

## Want to Contribute?

I welcome contributions! Here's how you can help:
1. Fork the project
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Make your changes
4. Submit them for review (`git push origin feature/YourFeature`)
5. Open a Pull Request

## Project Goals

I'm working to:
- Make eye disease detection more accessible
- Compare different AI approaches to find the best one
- Create a tool that's easy for others to use and build upon

Need help or have questions? Feel free to open an issue on my GitHub repository!
