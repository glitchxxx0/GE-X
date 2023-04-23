import cv2
import numpy as np
import tensorflow as tf
import webbrowser
import pyttsx3
import time
engine=pyttsx3.init('sapi5')
voices=engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
def speak(text):
    engine.say(text)
    engine.runAndWait()


# Load the pre-trained model from the .h5 file
model = tf.keras.models.load_model('action.h5')

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define a function to preprocess each frame of the video
def preprocess_frame(frame):
    # Perform any necessary preprocessing, such as resizing, normalization, etc.
    preprocessed_frame = cv2.resize(frame, (554, 30))
    preprocessed_frame = preprocessed_frame / 255.0  # normalize pixel values
    return preprocessed_frame

# Load the video
cap = cv2.VideoCapture(1)

# Loop through the frames of the video
while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        # Preprocess the frame
        preprocessed_frame = preprocess_frame(frame)
        
        # Reshape the preprocessed frame to have shape (1, 30, num_features)
        preprocessed_frame = np.reshape(preprocessed_frame, (1, 30, -1))
        
        # Make predictions using the preprocessed frame
        predictions = model.predict(preprocessed_frame)
        
        # Post-process the predictions as needed
        # ...
        if np.argmax(predictions)==1:
            speak("Netflix, Open Now!")
            webbrowser.open_new_tab('https://www.netflix.com/browse')
            time.sleep(5)
        elif np.argmax(predictions)==2:
            speak("Spotify, Open Now!")
            webbrowser.open_new_tab('https://open.spotify.com/')
            time.sleep(5)
        else:
            speak("Gesture Not Found")
            time.sleep(5) 
            
            
            

        # Display the frame
        cv2.imshow('Frame', frame)
        
        # Exit on ESC key press
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
