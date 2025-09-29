# Media Gesture Controller
Media Gesture Controller is a Python project that allows you to control any media player (e.g., Spotify, YouTube, VLC) using simple hand gestures captured by your webcam.
# Features
The application uses MediaPipe for accurate hand landmark detection and PyAutoGUI to simulate system media key presses, providing universal, hands-free media control.
| Gesture | Media Command | Action |
| :--- | :--- | :--- |
| **Thumb Up** | `volumeup` | Increase volume. |
| **Fist** | `volumedown` | Decrease volume. |
| **Open Palm** | `playpause` | Pause / Play. |
| **Victory Sign** (V-Sign) | `nexttrack` | Skip to the next track. |
# Setup and Installation
1. Requirements
   Python 3.x
   Webcam
2. Installation
   pip install opencv-python mediapipe numpy pyautogui
# How to Run
1. Save the provided code as a Python file (e.g., hand_controller.py).
2. Open the media player or video you wish to control.
3. Execute the script from your terminal:

Bash: python hand_controller.py

4. A window showing your webcam feed will open. Place your hand in the frame, and the program will start recognizing gestures.
5. To exit the program, press the Q key.

