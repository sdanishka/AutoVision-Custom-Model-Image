import streamlit as st
import numpy as np
import functions as f
import tensorflow as tf

# TITLE
st.markdown("<h1 style='text-align:center'>AutoVision: Custom Model Training</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center'>(Image Classification)</h4>", unsafe_allow_html=True)

# Set JSON Hyperparameter
st.session_state.hyperparameter = {}


def btnTrainingClicked():
    st.session_state.complete_data = isCompleteData()


def isCompleteData():
    all_sessions = list(st.session_state)
    image_sessions = []
    class_name_sessions = []
    for session in all_sessions:
        if "data_image_samples_" in session:
            image_sessions.append(session)
        elif "data_class_name_" in session:
            class_name_sessions.append(session)

    # Image Checking
    for image_session in image_sessions:
        if not st.session_state[image_session]:
            st.session_state.complete_data = False
            return False
    # Class Name Checking
    for class_name_session in class_name_sessions:
        if st.session_state[class_name_session] == "":
            st.session_state.complete_data = False
            return False

    st.session_state.complete_data = True
    return st.session_state.complete_data


def isUniqueClassName():
    isUnique = False
    all_sessions = list(st.session_state)
    class_name_sessions = []
    class_name_values = []
    for session in all_sessions:
        if "data_class_name_" in session:
            class_name_sessions.append(session)
    # Add Values Classname to List
    for class_name_session in class_name_sessions:
        class_name_values.append(st.session_state[class_name_session])
    # Unique Class Name Checking
    class_name_values = np.array(class_name_values)
    # Check if the lengths of the unique array and the original array are the same (if they are, then all elements
    # are unique)
    if len(np.unique(class_name_values)) == len(class_name_values):
        isUnique = True

    return isUnique


def setHyperparams(epochs, batch_size, lr):
    st.session_state.hyperparameter['epochs'] = epochs
    st.session_state.hyperparameter['batch_size'] = batch_size
    st.session_state.hyperparameter['lr'] = lr


col1, col2, col3, col4, col5 = st.columns(5, gap="large")
with col3:
    total_class_input = st.text_input("Number of Image Classes", key="data_num_class")

# DATA INPUT
if total_class_input:
    try:
        if int(total_class_input) < 2:
            st.warning('Please fill number of classes more than 2', icon="⚠️")
        else:

            for idx_class in range(int(total_class_input)):
                st.text_input("Fill Class Name", label_visibility='hidden', placeholder="Class Name",
                              key="data_class_name_{}".format(idx_class), on_change=isCompleteData)
                st.file_uploader("Add image to train", accept_multiple_files=True, key="data_image_samples_{}"
                                 .format(idx_class), on_change=isCompleteData, type=['jpg', 'jpeg', 'png', 'bmp'])

            # TRAINING
            col1, col2, col3 = st.columns(3, gap="large")
            # with col2:
            btn_training = col2.button("Train Model", type="primary")

            checkbox_advanced = col2.checkbox("Advanced")
            if checkbox_advanced:
                epochs_input = col2.number_input("Epochs", min_value=1, value=100)
                batch_size_input = col2.selectbox("Batch Size", (16, 32, 64, 128, 256, 512))
                lr_input = col2.number_input("Learning Rate", min_value=0.000001, value=0.001, step=0.001, format="%g")
                setHyperparams(epochs_input, batch_size_input, lr_input)

            if btn_training:
                # If the "Advanced" checkbox is not checked, then set default hyperparameters.
                if not checkbox_advanced:
                    # Set Default Parameter in Session
                    setHyperparams(epochs=1, batch_size=16, lr=0.0001)
                # If there is still data that has not been filled in.
                if isCompleteData():
                    if isUniqueClassName():
                        # Training process
                        model = f.trainingModel()
                    else:
                        col2.warning("Class Name must be Unique", icon='⚠️')
                else:
                    col2.warning("Data doesn't Complete", icon='⚠️')
    except Exception as e:
        print(e)
        st.warning('Please fill with number', icon="⚠️")

# print(st.session_state.isModelTrained)
try:
    if st.session_state.isModelTrained:
        try:
            model = tf.keras.models.load_model('trained_model.h5')
            f.displaySidebarResults(model)
        except Exception as e:
            print(e)
        except:
            st.warning("Input more images to get model evaluation", icon='⚠️')

        model = tf.keras.models.load_model('trained_model.h5')
        f.predictModel(model)
except:
    st.warning("Train model first to predict data", icon='ℹ️')

# st.write(st.session_state)
