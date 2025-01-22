import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from keras.models import load_model
from IPython.display import display, Javascript, Image
from google.colab.output import eval_js
from base64 import b64decode

# Load pre-trained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_model = load_model('/content/emotion_model.hdf5')

# Define emotion labels
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Initialize variables for statistics
total_faces = 0
total_emotions = {label: 0 for label in emotion_labels.values()}

# Function to detect faces and predict emotions
def detect_faces_and_emotions(frame):
    global total_faces, total_emotions

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Update total number of faces
    total_faces += len(faces)

    # Loop over detected faces
    for (x, y, w, h) in faces:
        # Extract face region from the frame
        face_roi = gray[y:y+h, x:x+w]

        # Resize face region to match model input size

        face_roi = cv2.resize(face_roi, (64, 64))

        # Normalize pixel values
        face_roi = face_roi / 255.0

        # Reshape image for model input
        face_roi = np.expand_dims(np.expand_dims(face_roi, -1), 0)

        # Predict emotion
        predicted_emotion = emotion_model.predict(face_roi)[0]

        # Get predicted emotion label
        emotion_label = emotion_labels[np.argmax(predicted_emotion)]

        # Draw bounding box around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display predicted emotion label
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Update total number of emotions
        total_emotions[emotion_label] += 1

    return frame

# Function to display statistics
def display_statistics(frame):
    global total_faces, total_emotions

    # Display total number of faces
    cv2.putText(frame, f'Total Faces: {total_faces}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display number of each emotion detected
    y_offset = 60
    for label, count in total_emotions.items():
        cv2.putText(frame, f'{label}: {count}', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30

    return frame

# Function to save the frame as an image
def save_image(frame, filename):
    cv2.imwrite(filename, frame)

# Main function to process video stream from webcam
def process_video_stream():
    # Open video capture device (webcam)
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the video stream
        ret, frame = cap.read()

        # Check if frame is successfully read
        if not ret:
            break

        # Detect faces and emotions in the frame
        frame_with_faces_and_emotions = detect_faces_and_emotions(frame)

        # Display statistics
        frame_with_statistics = display_statistics(frame_with_faces_and_emotions)

        # Display the frame with detected faces, emotions, and statistics
        cv2_imshow(frame_with_statistics)

        # Save the frame as an image
        save_image(frame_with_statistics, '/content/photo.jpg')

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture device and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Function to capture photo from webcam
def take_photo(filename='photo.jpg', quality=0.8):
    js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
    display(js)
    data = eval_js('takePhoto({})'.format(quality))
    binary = b64decode(data.split(',')[1])
    with open(filename, 'wb') as f:
        f.write(binary)
    return filename

# Main function to process captured photo
def process_captured_photo():
    try:
        filename = take_photo()
        print('Saved to {}'.format(filename))

        # Load the captured image
        frame = cv2.imread(filename)

        # Detect faces and emotions in the frame
        frame_with_faces_and_emotions = detect_faces_and_emotions(frame)

        # Display statistics
        frame_with_statistics = display_statistics(frame_with_faces_and_emotions)

        # Display the frame with detected faces, emotions, and statistics
        cv2_imshow(frame_with_statistics)
    except Exception as err:
        print(str(err))

# Run the main function
if __name__ == '__main__':
    choice = input("Choose an option:\n1. Capture Photo\nEnter your choice (1): ")
    if choice == '1':
        process_captured_photo()
    else:
        print("Invalid choice.")
