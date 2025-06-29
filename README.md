# SmartSorting 🍎🍌🥦

**Smart Sorting: Transfer Learning for Identifying Rotten Fruits and Vegetables**

This project predicts whether a fruit or vegetable is **fresh** or **rotten** using a deep learning model trained with transfer learning. The trained model (`model.h5`) is integrated into a Flask web application, allowing users to upload an image and instantly receive a prediction.

## 🚀 How to Run This Project

1. Install the required libraries:
```
pip install -r requirements.txt
```

2. Run the Flask application:
```
python app.py
```

3. Open your browser and go to:
```
http://127.0.0.1:5000/
```

4. Upload an image of a fruit or vegetable and get the result.

## 📁 Project Structure

```
SmartShorting/
├── app.py                → Flask web application
├── train_model.py        → Script used for training the model
├── model.h5              → Pre-trained model file
├── requirements.txt      → List of required Python libraries
├── templates/
│   └── index.html        → Frontend HTML file
├── static/
│   └── uploads/          → Folder where uploaded images are stored
```

## ⚠️ Notes

- The `model.h5` file is already trained and used for making predictions.
- The original training dataset (`train/` folder) is **not included** in this repository due to size limitations.
- This project was created as part of the **SmartBridge APSCHE Internship** program.

## 🛠️ Technologies Used

- Python
- TensorFlow & Keras (for training and prediction)
- Flask (for web interface)
- OpenCV & NumPy (for image processing)
- Matplotlib (for data visualization during training)

TEAM MEMBER/LEADER : DEEKSHITHA
