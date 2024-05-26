import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import load_img, img_to_array, array_to_img, to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
import base64


import streamlit as st
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def getLabelAndImagesArray():
    # Create a way to store information about labels that have been converted into numbers.
    st.session_state.label_data = {}
    labels_data = []
    images_data = []
    all_sessions = list(st.session_state)
    class_name_sessions = []
    images_sessions = []
    for session in all_sessions:
        if "data_class_name_" in session:
            class_name_sessions.append(session)
        elif "data_image_samples_" in session:
            images_sessions.append(session)
    
    class_name_sessions_sorted = sorted(class_name_sessions)
    images_sessions_sorted = sorted(images_sessions)

    label_number = 0
    for class_name_session, images_session in zip(class_name_sessions_sorted, images_sessions_sorted):
        class_name = st.session_state[class_name_session]
        st.session_state.label_data[label_number] = class_name
        for image_session in st.session_state[images_session]:
            # Add Label
            labels_data.append(label_number)
            # Add Image
            img = load_img(image_session, target_size=(256, 256))
            img_arr = img_to_array(img)
            images_data.append(img_arr)
        label_number+=1
    # Randomize to maximize the training process.
    images_data, labels_data = shuffle(images_data, labels_data)
    return np.array(images_data), np.array(labels_data)

def calculateConfusionMatrix(model, data_test_gen):
    labels_data = []
    num_batches = len(data_test_gen)

    # Get All Label Data
    for i in range(num_batches):
        images, labels = data_test_gen[i]
        for label in labels:
            labels_data.append(label)
    
    y_true = np.array(labels_data)
    y_true = np.argmax(y_true, axis=1)
    y_pred = model.predict(data_test_gen)
    # y_pred = json.dumps(y_pred.tolist()) # EagerTensor Error
    y_pred = np.argmax(y_pred, axis=1)
    cm_results = confusion_matrix(y_true, y_pred)
    return cm_results

def displayConfusionMatrix(cm):
    label_names = list(st.session_state.label_data.values()) 
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names).plot()
    return display


st.cache(suppress_st_warning=True)
def trainingModel():
    # Hyperparams
    epochs = st.session_state.hyperparameter["epochs"]
    batch_size = st.session_state.hyperparameter["batch_size"]
    lr = st.session_state.hyperparameter["lr"]
    
    # Data Input
    images_data, labels_data = getLabelAndImagesArray()
    num_label = len(np.unique(labels_data))

    # One-Hot Encoding Labels Data
    labels_data = to_categorical(labels_data, num_label)

    # Datagen
    datagen = ImageDataGenerator(
                featurewise_center=True,
                featurewise_std_normalization=True,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                validation_split=0.2
            )
    # Transform all data into the desired parameters from the data generator.
    datagen.fit(images_data)

    train_data_gen = datagen.flow(images_data, labels_data, batch_size=batch_size, subset='training')
    test_data_gen = datagen.flow(images_data, labels_data, batch_size=batch_size, subset='validation')

    # Model CNN

    # Creating a MobileNetV2 model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    # Freezing all layers in the base model to prevent them from being changed during training.
    for layer in base_model.layers[:-3]:
        layer.trainable = False
    # Adding a global average pooling layer and a classification output layer.
    base_model_output = base_model.output
    flatten = layers.GlobalAveragePooling2D()(base_model_output)
    out = layers.Dense(num_label, activation='softmax')(flatten)
    model = models.Model(inputs=base_model.input, outputs=out)
    # Configure CNN Model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
                loss='categorical_crossentropy', 
                metrics=['accuracy'],
                )
    # Training Data
    model.fit(train_data_gen, epochs=epochs, batch_size=batch_size)
    
    # Calculate Confusion Matrix
    cm_results = calculateConfusionMatrix(model, test_data_gen)

    st.session_state.cm_results = cm_results
    st.session_state.isModelTrained = True
    model.save('trained_model.h5')
    return model

def displaySidebarResults(model):
    cm = st.session_state.cm_results
    # Display Confusion Matrix
    display_cm = displayConfusionMatrix(cm)
    accuracy_per_class = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    accuracy_per_class = accuracy_per_class.diagonal()
    # create sidebar
    st.sidebar.markdown("<h1 style='text-align:center'>Confusion Matrix Result</h1>", unsafe_allow_html=True)
    st.set_option('deprecation.showPyplotGlobalUse', False) # Agar warning tidak keluar
    st.sidebar.pyplot(plt.show())
    st.sidebar.markdown("<h1 style='text-align:center'>Accuracy Result per Class</h1>", unsafe_allow_html=True)
    for idx, label in enumerate(list(st.session_state.label_data.values())):
        if np.isnan(accuracy_per_class[idx]):
            st.sidebar.write("Class: {} - Accuracy: No sample data available".format(label, accuracy_per_class[idx]))
        else:
            st.sidebar.write("Class: {} - Accuracy: {}".format(label, accuracy_per_class[idx]))

    file_path = 'trained_model.h5'
    model = model.to_json().encode('utf-8')
    encoded_h5 = base64.b64encode(model).decode('utf-8')
    href = f'<a href="data:file/h5;base64,{encoded_h5}" download="{file_path}">Download Model .h5</a>'
    st.sidebar.markdown(href, unsafe_allow_html=True)

def getPredictedImage(model):
    img = load_img(st.session_state.data_image_predict, target_size=(256, 256))
    img_arr = img_to_array(img)[np.newaxis, ...]
    result = model.predict(img_arr)
    return result

def predictModel(model):
    # PROSES PREDICTION
    st.markdown("<h1 style='text-align:center;'> -----Try to Predict Image----- </h1>", unsafe_allow_html=True)
    image_predict = st.file_uploader("Add image to predict", accept_multiple_files=False, key="data_image_predict", type=['jpg', 'jpeg', 'png', 'bmp'])
    btn_predict_image = st.button("Predict", type="primary")
    if btn_predict_image:
        if image_predict:
            st.markdown("<h4>Image Predict</h4>", unsafe_allow_html=True)
            st.image(image_predict)
            st.markdown("<h4>Results</h4>", unsafe_allow_html=True)
            result = getPredictedImage(model)
            for probability, label in zip(result[0], list(st.session_state.label_data.values())):
                st.write("Class: %s - Probability: %.3f" % (label, probability*100))
        else:
            st.warning('Fill the image first', icon="⚠️")
