import pyaudio
import wave
from deepgram import DeepgramClient, PrerecordedOptions
import os
import google.generativeai as genai
import requests
import numpy as np
import cv2
import time
import sys

# Setting API keys
DEEPGRAM_API_KEY = 'DEEPGRAM_API_KEY'
GOOGLE_API_KEY = 'GOOGLE_API_KEY'
HF_API_TOKEN = 'HF_API_TOKEN'

# Configuring Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(
    'gemini-1.5-flash',
    system_instruction="You are a friendly AI assistant who loves to chat casually in English and can describe images in one short sentence."
)
chat = model.start_chat(history=[])

# Function to record audio with silence detection
def record_audio(filename):
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    rate = 16000
    silence_threshold = 500
    silence_duration = 3
    
    p = pyaudio.PyAudio()
    stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
    print("Recording... (Speak now, stops after 3 seconds of silence)")
    
    frames = []
    silent_chunks = 0
    max_silent_chunks = int(silence_duration * rate / chunk)
    
    while True:
        data = stream.read(chunk, exception_on_overflow=False)
        frames.append(data)
        audio_data = np.frombuffer(data, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_data**2)) if len(audio_data) > 0 else 0
        if rms < silence_threshold:
            silent_chunks += 1
        else:
            silent_chunks = 0
        if silent_chunks >= max_silent_chunks:
            break
    
    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

# Function to convert speech to text
def speech_to_text(filename):
    deepgram = DeepgramClient(DEEPGRAM_API_KEY)
    with open(filename, 'rb') as audio:
        buffer_data = audio.read()
    options = PrerecordedOptions(model="general", punctuate=True)
    response = deepgram.listen.rest.v("1").transcribe_file(
        {"buffer": buffer_data, "mimetype": "audio/wav"}, options
    )
    return response["results"]["channels"][0]["alternatives"][0]["transcript"]

# Function to convert text to speech
def text_to_speech(text):
    url = "https://api.deepgram.com/v1/speak?model=aura-asteria-en"
    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "application/json"}
    data = {"text": text}
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        with open('response.mp3', 'wb') as f:
            f.write(response.content)
        os.system('mpg123 response.mp3')
    else:
        print(f"Failed to generate speech: {response.status_code} - {response.text}")

# Function to capture and crop image from webcam
def capture_and_crop_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None, "Sorry, I couldn't access your webcam!"
    
    print("Showing webcam... Hold your drawing up inside the red rectangle and wait 10 seconds.")
    
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None, "Failed to capture image from webcam!"
    frame_height, frame_width = frame.shape[:2]

    rect_width = 400
    rect_height = 300
    rect_x = (frame_width - rect_width) // 2
    rect_y = (frame_height - rect_height) // 2

    start_time = time.time()
    photo_taken = False

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None, "Failed to capture image from webcam!"

        elapsed_time = time.time() - start_time
        cv2.rectangle(frame, (rect_x, rect_y), 
                     (rect_x + rect_width, rect_y + rect_height), 
                     (0, 0, 255), 2)
        cv2.imshow("Webcam", frame)

        if elapsed_time >= 10 and not photo_taken:
            cropped = frame[rect_y:rect_y + rect_height, rect_x:rect_x + rect_width]
            image_path = 'captured_image.png'
            cv2.imwrite(image_path, cropped)
            print("Image saved at 10 seconds!")
            photo_taken = True
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return image_path, "I captured your drawing!"

# Function to process image for description
def process_image(image_path):
    with open(image_path, 'rb') as img_file:
        img_data = img_file.read()
    response = model.generate_content([
        {"mime_type": "image/png", "data": img_data},
        {"text": "Describe the drawing in this image in one short sentence."}
    ])
    return response.text

# Function to generate cartoon image from text
def text_to_image(prompt, output_image_path):
    API_URL = "https://api-inference.huggingface.co/models/XLabs-AI/flux-RealismLora"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {"num_inference_steps": 40, "guidance_scale": 8}
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        with open(output_image_path, "wb") as f:
            f.write(response.content)
        print(f"Cartoon image saved at {output_image_path}")
        return True
    else:
        print(f"Error generating cartoon: {response.status_code} - {response.text}")
        return False

# Main loop
print("Hey! I'm waiting for you to say something. Say 'goodbye' to stop or 'cartoon' to make a cartoon.")
while True:
    record_audio('user_input.wav')
    user_text = speech_to_text('user_input.wav')
    print(f"You said: {user_text}")
    
    # If nothing was said or only silence was detected
    if not user_text.strip():
        print("I didn't hear anything. Please try again!")  # Only print, no speech
        continue  # Wait for next input
    
    # If user said "goodbye"
    if user_text.lower() == "goodbye":
        text_to_speech("Goodbye! See you later!")
        sys.exit(0)  # Explicitly exit the program
    
    # If user said "cartoon"
    elif "cartoon" in user_text.lower():
        image_path, capture_response = capture_and_crop_image()
        text_to_speech(capture_response)
        if image_path:
            description = process_image(image_path)
            print(f"Gemini description: {description}")
            text_to_speech(description)
            cartoon_path = "cartoon_output.jpg"
            if text_to_image(description, cartoon_path):
                text_to_speech("I created a cartoon from the description!")
            else:
                text_to_speech("Sorry, I couldn't create the cartoon.")
    
    # If user said something else
    else:
        gemini_response = chat.send_message(user_text)
        print(f"Gemini says: {gemini_response.text}")
        text_to_speech(gemini_response.text)