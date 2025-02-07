import tkinter as tk
import speech_recognition as sr
import threading
from PIL import Image, ImageTk
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import customtkinter
from textblob import TextBlob
import openai

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("blue")

openai.api_key = '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'

face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
classifier = load_model(r'model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Create the main Tkinter window
root = customtkinter.CTk()
root.title("Personality Detector")
root.geometry('1000x1000')

frame_1 = customtkinter.CTkFrame(master=root)
frame_1.pack(pady=20, padx=60, fill="both", expand=True)

# Create a label for displaying the video
video_label = customtkinter.CTkLabel(master=frame_1)
video_label.pack(anchor='nw')  # Use 'nw' anchor to place the video frame in the top left corner

# Create a label for displaying the emotion
emotion_label = customtkinter.CTkLabel(master=frame_1, text="Emotions: ", font=("Helvetica", 20))
emotion_label.place(x=700, y=0)

# Adding microphone switch
mic_switch_value = tk.BooleanVar()
mic_switch = customtkinter.CTkSwitch(master=frame_1, text="Microphone", variable=mic_switch_value)
mic_switch.place(x=10, y=400)

voice_data = customtkinter.CTkTextbox(master=frame_1, width=500, height=100)
voice_data.place(x=10, y=450)

sentiment_label = customtkinter.CTkLabel(master=frame_1, text="Sentiment: ", font=("Helvetica", 20))
sentiment_label.place(x=10, y=600)

gpt_label = customtkinter.CTkLabel(master=frame_1, text="GPT Response: ", font=("Helvetica", 20))
gpt_label.place(x=10, y=700)

classifier = load_model('model.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def toggle_microphone():
    if mic_switch_value.get():
        threading.Thread(target=listen_to_speech).start()
    else:
        voice_data.delete(1.0, tk.END)


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

            get_gpt_response(voice_text)

        except sr.WaitTimeoutError:
            voice_data.delete(1.0, tk.END)
            voice_data.insert(tk.END, "Listening timed out.")
        except sr.UnknownValueError:
            voice_data.delete(1.0, tk.END)
            voice_data.insert(tk.END, "Could not understand audio.")
        except sr.RequestError:
            voice_data.delete(1.0, tk.END)
            voice_data.insert(tk.END, "Network error occurred.")


def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity

    if sentiment > 0:
        sentiment_label.configure(text="Sentiment: Positive")
    elif sentiment < 0:
        sentiment_label.configure(text="Sentiment: Negative")
    else:
        sentiment_label.configure(text="Sentiment: Neutral")


def get_gpt_response(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": text}]
        )
        gpt_response = response['choices'][0]['message']['content']
        gpt_label.configure(text="GPT Response: " + gpt_response)
    except Exception as e:
        gpt_label.configure(text="GPT Error: " + str(e))

def update_frame():
    _, frame = cap.read()
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

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=im)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk, text="")

    root.after(15, update_frame)


mic_switch.configure(command=toggle_microphone)

cap = cv2.VideoCapture(0)
update_frame()

root.mainloop()

cap.release()
cv2.destroyAllWindows()
