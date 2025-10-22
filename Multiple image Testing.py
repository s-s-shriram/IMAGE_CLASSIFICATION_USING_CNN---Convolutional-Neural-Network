#NOTE:
"""Before running below code need to execute the main code of Model Training
This is code used for reducing the Time efficiency to avoid again and training the model and testing"""

# Upload your test image
uploaded = files.upload()  # Choose any image file

for filename in uploaded.keys():
    path = filename
    # Load and preprocess image
    img = load_img(path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    # Show image with predicted class
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}")
    plt.axis('off')
    plt.show()
