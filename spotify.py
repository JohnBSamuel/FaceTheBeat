import tkinter as tk
import speech_recognition as sr
import threading
from PIL import Image, ImageTk
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import customtkinter
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from textblob import TextBlob

# Spotify API credentials
SPOTIPY_CLIENT_ID = '291abb5330f142f49fca6721455c9564'
SPOTIPY_CLIENT_SECRET = 'dd12620eb5e948b88c83ce338a275d95'
SPOTIPY_REDIRECT_URI = 'http://localhost:8000/callback'

# Set up Spotify OAuth and create a Spotify object
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID,
                                               client_secret=SPOTIPY_CLIENT_SECRET,
                                               redirect_uri=SPOTIPY_REDIRECT_URI,
                                               scope="user-read-playback-state,user-modify-playback-state"))

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("blue")

# Load models and classifiers
face_cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
classifier = load_model(r'model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

root = customtkinter.CTk()
root.title("Personality Detector")
root.geometry('1000x1000')

frame_1 = customtkinter.CTkFrame(master=root)
frame_1.pack(pady=20, padx=60, fill="both", expand=True)

video_label = customtkinter.CTkLabel(master=frame_1)
video_label.pack(anchor='nw')

emotion_label = customtkinter.CTkLabel(master=frame_1, text="Emotions: ", font=("Helvetica", 20))
emotion_label.place(x=700, y=0)

mic_switch_value = tk.BooleanVar()
mic_switch = customtkinter.CTkSwitch(master=frame_1, text="Microphone", variable=mic_switch_value)
mic_switch.place(x=10, y=400)

voice_data = customtkinter.CTkTextbox(master=frame_1, width=500, height=100)
voice_data.place(x=10, y=450)

sentiment_label = customtkinter.CTkLabel(master=frame_1, text="Sentiment: ", font=("Helvetica", 20))
sentiment_label.place(x=10, y=600)

mic_active = False

# Toggle microphone
def toggle_microphone():
    global mic_active
    mic_active = mic_switch_value.get()
    if mic_active:
        threading.Thread(target=listen_to_speech).start()
    else:
        voice_data.delete(1.0, tk.END)

# Speech recognition
def listen_to_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        voice_data.delete(1.0, tk.END)
        voice_data.insert(tk.END, "Listening...")
        try:
            audio = recognizer.listen(source, timeout=5)
            voice_data.delete(1.0, tk.END)
            voice_text = recognizer.recognize_google(audio)
            voice_data.insert(tk.END, voice_text)
            analyze_sentiment(voice_text)
        except sr.WaitTimeoutError:
            voice_data.delete(1.0, tk.END)
            voice_data.insert(tk.END, "Listening timed out.")
        except sr.UnknownValueError:
            voice_data.delete(1.0, tk.END)
            voice_data.insert(tk.END, "Could not understand audio.")
        except sr.RequestError:
            voice_data.delete(1.0, tk.END)
            voice_data.insert(tk.END, "Network error occurred.")

# Sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        sentiment_label.configure(text="Sentiment: Positive")
    elif sentiment < 0:
        sentiment_label.configure(text="Sentiment: Negative")
    else:
        sentiment_label.configure(text="Sentiment: Neutral")

# Music recommendations
def recommend_music_based_on_emotion(emotion):
    emotion_genre_map = {
        'Angry': 'rock',
        'Disgust': 'grunge',
        'Fear': 'ambient',
        'Happy': 'pop',
        'Neutral': 'indie',
        'Sad': 'acoustic',
        'Surprise': 'electronic'
    }
    genre = emotion_genre_map.get(emotion, 'pop')
    results = sp.recommendations(seed_genres=[genre], limit=10)
    tracks = [f"{track['name']} by {track['artists'][0]['name']}" for track in results['tracks']]
    return tracks

# Update video frame
def update_frame():
    ret, frame = cap.read()
    if not ret:
        return  # If no frame is read, exit the function

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        prediction = classifier.predict(roi)[0]
        label = emotion_labels[prediction.argmax()]
        emotion_label.configure(text="Emotions: " + label)

        # Fetch music recommendations
        recommended_tracks = recommend_music_based_on_emotion(label)
        voice_data.delete(1.0, tk.END)
        voice_data.insert(tk.END, "Recommended Music:\n" + "\n".join(recommended_tracks))

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Convert frame to Image
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=im)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk, text="")

    root.after(15, update_frame)

# Bind microphone switch to function
mic_switch.configure(command=toggle_microphone)

# Start video capture
cap = cv2.VideoCapture(0)
update_frame()

root.mainloop()

cap.release()
cv2.destroyAllWindows()
