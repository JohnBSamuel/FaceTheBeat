import tkinter as tk
import threading
from PIL import Image, ImageTk
import cv2
import numpy as np
import customtkinter
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import time  # for tracking time between updates

# Initialize customtkinter settings
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("blue")

# Load the emotion detection model and face cascade
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Set up the main GUI window
root = customtkinter.CTk()
root.title("Emotion Detector with Stories")
root.geometry('1000x800')

# Create GUI layout frames and widgets
frame_1 = customtkinter.CTkFrame(master=root)
frame_1.pack(pady=20, padx=60, fill="both", expand=True)

video_label = customtkinter.CTkLabel(master=frame_1)
video_label.pack(anchor='nw')

emotion_label = customtkinter.CTkLabel(master=frame_1, text="Emotions: ", font=("Helvetica", 20))
emotion_label.place(x=700, y=0)

story_text = customtkinter.CTkTextbox(master=frame_1, width=500, height=300)
story_text.place(x=10, y=450)
story_text.insert(tk.END, "Your story will appear here based on detected emotion...")

stories = {
    "Angry": [
        "The Unbreakable Knight: In a kingdom tormented by marauding forces, Sir Gareth is consumed with rage over the suffering of his people. Through his journey across perilous lands, he encounters mythical creatures, each testing his anger and his resolve. Gradually, he learns that anger can be channeled into strength and courage, becoming a force for good as he leads his kingdom to victory.",
        "The Storm Within: Maya, an environmental activist, witnesses the destruction of her beloved forest due to corporate greed. Her anger drives her to confront powerful foes, challenging her beliefs. As her journey unfolds, she realizes that true change requires more than fury—it requires resilience, strategy, and empathy.",
        "The Red Sea: Arjun is a warrior forced to battle against his kin in a civil war driven by pride and revenge. Amidst the turmoil, his anger toward those who tore his homeland apart fuels his strength. However, through facing his inner rage, he learns that anger, unchecked, can also lead to his downfall, and peace lies in forgiveness.",
        "The Fiery Inventor: Livia, an inventor, becomes enraged when her groundbreaking technology is stolen. Driven by this fury, she builds an even better creation, only to face the choice of either crushing her competitors or using her power to foster positive change. Her journey reflects how anger can spark innovation but must be balanced by responsibility."
    ],
    "Disgust": [
        "The Purifier of the Land: In a once-idyllic valley, the town of Celeste is overcome by pollution from nearby factories. Annabelle, a resident disgusted by the state of her homeland, sets out on a quest to rid the land of the toxins. Her journey is both external and internal as she cleanses the world and her own feelings of disdain.",
        "The Glass House: Laura is a detective who despises dishonesty and corruption. Her latest case, investigating a high-profile corporate scandal, forces her to navigate a web of deceit. Her disgust with the truth leads her to expose hidden truths, ultimately questioning her own assumptions about morality.",
        "The Scorned Chef: Marcello, a renowned chef, is disgusted by how his art is trivialized by food critics and influencers. His journey through culinary arts, innovation, and defiance of popular trends is fueled by his distaste for shallow opinions, ultimately bringing him to rediscover the true purpose of his craft.",
        "The City’s Conscience: In a future dystopian city, Elara is horrified by the injustices of a corrupt government. Driven by her disgust, she joins an underground rebellion. Her disgust becomes a catalyst, as she inspires others to rise up and reclaim their rights, transforming disgust into a weapon for social change."
    ],
    "Fear": [
        "The Cave of Shadows: On the edge of a forbidden forest lies the Cave of Shadows, a place rumored to hold an ancient evil. Young adventurer Ethan, driven by the fear of the unknown, ventures inside to uncover its secrets. His journey is filled with terrifying encounters, yet he learns that courage is not the absence of fear, but facing it head-on.",
        "The Haunted Library: Rosa, a quiet librarian, discovers strange occurrences in her old library. As she delves deeper, she encounters ghostly apparitions, whispering voices, and books that seem to open on their own. Her fear turns to curiosity as she uncovers the library's secrets, discovering that her bravery was stronger than she believed.",
        "The Skyfall Kingdom: Princess Lyra lives in a floating kingdom, fearful of the mysterious 'Skyfall' that has plagued her people. As she journeys to the heart of the kingdom, she confronts her fear of falling and faces a hidden prophecy. She learns that only by facing the unknown can she protect her kingdom from doom.",
        "Through the Mirror: Liam, a reclusive artist, becomes obsessed with a mirror that shows his darkest fears. His fear escalates until he’s forced to confront his hidden anxieties. Through this haunting experience, he transforms his fear into creativity, realizing that art can be a powerful way to face inner demons."
    ],
    "Happy": [
        "The Sunlit Festival: In the village of Haven, residents come together annually for the Sunlit Festival, a celebration of community and joy. As the festival unfolds, friendships blossom, love stories begin, and old feuds are put to rest. The spirit of happiness brings everyone closer, creating memories that last a lifetime.",
        "The Garden of Wonders: A young boy, Leo, discovers a hidden garden full of magical plants and friendly creatures. Each day he visits, the garden shows him something new and wondrous, filling his heart with happiness. Through his experiences, he learns that true happiness lies in cherishing nature’s simple and magical moments.",
        "The Reunion of Hearts: After years of separation, childhood friends reunite in their hometown. Their shared memories bring back laughter and joy as they relive their best times together. The power of friendship fills them with warmth, reminding them that happiness is deeply connected to shared history and companionship.",
        "The Painter’s Dream: Mira, a struggling artist, finds happiness in painting scenes from her dreams. Her vibrant work catches the eye of a gallery owner, and soon her art brings joy to others. Her story reflects how following one’s passion can lead to happiness and even inspire it in others."
    ],
    "Neutral": [
        "The Observer: In a bustling city, an unnamed observer sits daily in a quiet café, watching people go by. He sees stories unfold—romances, family reunions, and silent gestures of kindness. Though his life is unremarkable, he finds contentment in observing these everyday stories, realizing that life’s beauty is in the simple moments.",
        "The Wanderer’s Notebook: Amelia is a traveler who records her thoughts in a journal as she explores various cities. Her observations on the mundane aspects of different cultures help her appreciate life’s small wonders. Her story reminds us that neutrality can be a form of peace, a way to find beauty without needing grand events.",
        "The Museum of Echoes: Jacob, a museum curator, spends his days restoring artifacts from the past. Although his life may seem routine, he finds meaning in preserving history. His quiet passion reveals that neutrality isn’t dullness—it’s a form of respect for life’s quieter, timeless values.",
        "The Silent Morning: A young woman spends a morning on her balcony, observing the world awaken. Birds sing, neighbors greet each other, and the sun rises slowly. Her neutral observation of the world’s beauty teaches her that sometimes, happiness is simply being present in the moment."
    ],
    "Sad": [
        "The Willow Tree: A young man returns to his hometown, where he visits a large willow tree that has stood for generations. He reflects on his lost loved ones and the changes in his life. The tree becomes a symbol of resilience, holding memories of the past, even as time moves forward.",
        "The Farewell Letter: Emma, a grandmother, writes a letter to her late husband, reminiscing on their life together. As she pours her heart onto paper, she grapples with her grief and loneliness. Through her story, she finds solace, realizing that while sadness lingers, love remains eternal.",
        "The Forgotten Melody: Leo, a former pianist, has lost his passion for music after a great personal loss. One day, he hears a melody that reignites memories of joy and sorrow. Slowly, he rediscovers his love for music, learning that embracing sadness can be part of the journey to healing.",
        "The Last Lantern: On a night of remembrance, a village releases lanterns into the sky for their departed loved ones. Each lantern carries a story of loss, sadness, and hope. Through the ritual, the villagers realize that even in sadness, there is a shared light that connects them, giving them strength."
    ],
    "Surprise": [
        "The Secret Garden Path: Sarah stumbles upon an overgrown path in her grandmother’s garden. Following it, she discovers a hidden world filled with magical creatures and enchanted flowers. The unexpected beauty reminds her that life is full of wonderful surprises waiting to be found.",
        "The Unlikely Heirloom: While cleaning her attic, Eliza finds a mysterious key that leads her to a hidden family treasure. Along the way, she uncovers unexpected stories about her ancestors, learning that surprises can reveal roots and family histories that shape who we are.",
        "The Mysterious Island: A group of friends finds themselves stranded on an uncharted island. As they explore, they uncover ancient ruins, hidden caves, and rare animals. Each discovery fills them with wonder, reminding them that life’s best moments are often unplanned.",
        "The Stranger’s Gift: On her way to work, Mia encounters a stranger who gifts her a beautifully crafted music box. She soon learns that the music box holds a secret message, leading her on a journey through the city. The experience shows her the joy that unexpected kindness and surprises can bring."
    ]
}

# Variables for tracking the current emotion and time
current_emotion = None
last_update_time = 0
story_indices = {emotion: 0 for emotion in emotion_labels}

# Function to update the video frame and story text
def update_frame():
    global current_emotion, last_update_time

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

        # Update story only if emotion has changed or 60 seconds have passed
        if current_emotion != label or (time.time() - last_update_time > 60):
            current_emotion = label
            last_update_time = time.time()  # reset timer for 60 seconds
            emotion_label.configure(text="Emotion: " + label)

            # Get the story index for the current emotion and display the story
            index = story_indices[current_emotion]
            story_text.delete(1.0, tk.END)
            story_text.insert(tk.END, stories[current_emotion][index])

            # Update the story index for next time
            story_indices[current_emotion] = (index + 1) % len(stories[current_emotion])

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

# Start the video capture and frame updates
cap = cv2.VideoCapture(0)
update_frame()

root.mainloop()
cap.release()
cv2.destroyAllWindows()