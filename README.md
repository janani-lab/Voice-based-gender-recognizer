# Voice-based Gender Recognizer

A Python project to detect and classify gender based on voice samples. This repository provides tools for extracting features from audio files and applying machine learning models to predict gender from voice data.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)

## Features

- Extracts relevant features from voice samples (e.g., MFCC, pitch, etc.)
- Trains machine learning models for gender classification
- Supports evaluation and inference on new audio samples
- 100% Python-based implementation

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/janani-lab/Voice-based-gender-recognizer.git
   cd Voice-based-gender-recognizer
   ```

> **Note:** Make sure you have Python 3.7 or higher installed.

## Usage

1. Prepare your dataset of voice samples in `.wav` format.
2. Run the feature extraction and training scripts.  
   Example:
   ```bash
   python extract_features.py --input data/ --output features.csv
   python train_model.py --features features.csv --model model.pkl
   ```
3. To predict gender from a new audio sample:
   ```bash
   python predict.py --model model.pkl --audio sample.wav
   ```

## Dataset

You can use publicly available voice datasets such as:
- [Kaggle Gender Recognition by Voice](https://www.kaggle.com/datasets/primaryobjects/voicegender)

Or your own `.wav` files. Place them in the `data/` directory.

## How It Works

1. **Feature Extraction:** Extracts voice features (MFCCs, spectral features, etc.) from audio data.
2. **Model Training:** Trains a classifier (e.g., SVM, Random Forest) on the extracted features.
3. **Prediction:** Classifies new voice samples as male or female based on the trained model.


## Contributing

Contributions are welcome! Please open issues or submit pull requests for features, bug fixes, or improvements.

## License

This project is licensed under the GPL-3.0 License. See the [LICENSE](LICENSE) file for details.
