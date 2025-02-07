import tkinter as tk
import threading
from PIL import Image, ImageTk
import cv2
import numpy as np
import customtkinter
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from textblob import TextBlob
import time
import openai

# Set your OpenAI API key
openai.api_key = "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"  # Replace with your actual OpenAI API key

# Initialize the customtkinter settings
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("blue")

# Load the emotion detection model and face cascade
face_cascade = cv2.CascadeClassifier('/Users/johnsamuel/Desktop/Emotion_Detection_CNN/Emotion_Detection_CNN/haarcascade_frontalface_default.xml')
classifier = load_model('/Users/johnsamuel/Desktop/Emotion_Detection_CNN/Emotion_Detection_CNN/model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Set up the main GUI window
root = customtkinter.CTk()
root.title("Emotion Detector")
root.geometry('1000x800')

# Create GUI layout frames and widgets
frame_1 = customtkinter.CTkFrame(master=root)
frame_1.pack(pady=20, padx=60, fill="both", expand=True)

video_label = customtkinter.CTkLabel(master=frame_1)
video_label.pack(anchor='nw')

emotion_label = customtkinter.CTkLabel(master=frame_1, text="Emotions: ", font=("Helvetica", 20))
emotion_label.place(x=700, y=0)

story_textbox = customtkinter.CTkTextbox(master=frame_1, width=250, height=300)
story_textbox.place(x=700, y=50)

mic_switch_value = tk.BooleanVar()
mic_switch = customtkinter.CTkSwitch(master=frame_1, text="Microphone", variable=mic_switch_value)
mic_switch.place(x=10, y=400)

voice_data = customtkinter.CTkTextbox(master=frame_1, width=500, height=100)
voice_data.place(x=10, y=450)

# Timer variables to control story update frequency
last_story_time = 0  # Track the last time the story was updated
story_update_interval = 1  # Time in seconds between story updates

# Function to generate a story using OpenAI API based on the detected emotion
def generate_dynamic_story(emotion):
    prompt = f"Write a creative short story that conveys a feeling of {emotion}."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use the gpt-3.5-turbo model
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        story = response.choices[0].message['content'].strip()
        return story
    except openai.error.OpenAIError as e:
        print(f"Error generating story: {e}")
        return "Could not generate story due to an error."

# Function to perform sentiment analysis using TextBlob
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        sentiment_text = "Positive"
    elif sentiment < 0:
        sentiment_text = "Negative"
    else:
        sentiment_text = "Neutral"
    return sentiment_text

# Function to update the video frame and detect emotion
def update_frame():
    global last_story_time  # Access the timer variable
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        prediction = classifier.predict(roi)[0]
        label = emotion_labels[prediction.argmax()]
        emotion_label.configure(text="Emotions: " + label)

        # Update the story only if enough time has passed
        current_time = time.time()
        if current_time - last_story_time >= story_update_interval:
            story = generate_dynamic_story(label)
            story_textbox.delete(1.0, tk.END)
            story_textbox.insert(tk.END, story)
            last_story_time = current_time  # Reset the timer

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Convert the frame to a PhotoImage and display it
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=im)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk, text="")

    # Schedule the next frame update
    root.after(15, update_frame)

# Function to listen to the microphone and analyze text sentiment
def listen_to_speech():
    import speech_recognition as sr
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        voice_data.delete(1.0, tk.END)  # Clear previous text
        voice_data.insert(tk.END, "Listening...")
        try:
            audio = recognizer.listen(source, timeout=5)
            voice_data.delete(1.0, tk.END)  # Clear previous text
            voice_text = recognizer.recognize_google(audio)
            voice_data.insert(tk.END, voice_text)

            # Perform sentiment analysis on the recognized text
            sentiment = analyze_sentiment(voice_text)
            print("Sentiment:", sentiment)

        except sr.WaitTimeoutError:
            voice_data.delete(1.0, tk.END)
            voice_data.insert(tk.END, "Listening timed out.")
        except sr.UnknownValueError:
            voice_data.delete(1.0, tk.END)
            voice_data.insert(tk.END, "Could not understand audio.")
        except sr.RequestError:
            voice_data.delete(1.0, tk.END)
            voice_data.insert(tk.END, "Network error occurred.")

# Function to toggle microphone listening
def toggle_microphone():
    if mic_switch_value.get():
        threading.Thread(target=listen_to_speech).start()
    else:
        voice_data.delete(1.0, tk.END)

# Bind the microphone switch to toggle_microphone function
mic_switch.configure(command=toggle_microphone)

# Start the video capture and frame updates
cap = cv2.VideoCapture(0)
update_frame()

root.mainloop()
cap.release()
cv2.destroyAllWindows()