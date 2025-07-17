import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk
import threading
import time
import math

class VirtualKeyboard:
    def __init__(self):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Keyboard layout
        self.keys = [
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
            ['Z', 'X', 'C', 'V', 'B', 'N', 'M'],
            ['SPACE', 'BACKSPACE', 'ENTER']
        ]
        
        # Key positions and sizes
        self.key_positions = {}
        self.key_size = (80, 60)
        self.key_margin = 10
        
        # Interaction variables
        self.clicked_key = None
        self.click_threshold = 30
        self.last_click_time = 0
        self.click_cooldown = 0.5
        
        # Text output
        self.typed_text = ""
        
        # Setup GUI
        self.setup_gui()
        self.calculate_key_positions()
        
        # Start camera thread
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
        
    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Virtual Hand-Controlled Keyboard")
        self.root.geometry("800x600")
        
        # Text display area
        text_frame = ttk.Frame(self.root)
        text_frame.pack(pady=10, padx=10, fill='x')
        
        ttk.Label(text_frame, text="Typed Text:").pack(anchor='w')
        self.text_display = tk.Text(text_frame, height=5, width=80)
        self.text_display.pack(fill='x')
        
        # Status display
        self.status_label = ttk.Label(self.root, text="Status: Initializing...")
        self.status_label.pack(pady=5)
        
        # Instructions
        instructions = """
        Instructions:
        1. Make sure your hand is visible in the camera
        2. Point your index finger at a key
        3. Bring your thumb close to your index finger to "click"
        4. Keep your hand steady for better detection
        """
        ttk.Label(self.root, text=instructions, justify='left').pack(pady=10)
        
        # Keyboard frame
        self.keyboard_frame = ttk.Frame(self.root)
        self.keyboard_frame.pack(pady=20)
        
        # Create keyboard buttons
        self.key_buttons = {}
        for row_idx, row in enumerate(self.keys):
            row_frame = ttk.Frame(self.keyboard_frame)
            row_frame.pack(pady=2)
            
            for col_idx, key in enumerate(row):
                if key == 'SPACE':
                    btn = ttk.Button(row_frame, text=key, width=20)
                elif key in ['BACKSPACE', 'ENTER']:
                    btn = ttk.Button(row_frame, text=key, width=12)
                else:
                    btn = ttk.Button(row_frame, text=key, width=8)
                
                btn.pack(side='left', padx=2)
                self.key_buttons[key] = btn
        
        # Quit button
        ttk.Button(self.root, text="Quit", command=self.quit_app).pack(pady=10)
        
    def calculate_key_positions(self):
        # This will be updated based on camera frame coordinates
        pass
    
    def get_distance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def is_finger_clicking(self, landmarks):
        # Get thumb tip and index finger tip
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # Calculate distance between thumb and index finger
        distance = self.get_distance(
            [thumb_tip.x, thumb_tip.y],
            [index_tip.x, index_tip.y]
        )
        
        # Convert to pixel distance (approximate)
        pixel_distance = distance * 640  # Assuming 640px width
        
        return pixel_distance < self.click_threshold
    
    def get_key_at_position(self, x, y, frame_width, frame_height):
        # Convert camera coordinates to key detection
        # This is a simplified version - you might want to make this more sophisticated
        
        # Define key regions based on screen position
        key_regions = {
            # Row 1 (numbers)
            '1': (0, 0.1, 0.1, 0.2), '2': (0.1, 0.2, 0.1, 0.2), '3': (0.2, 0.3, 0.1, 0.2),
            '4': (0.3, 0.4, 0.1, 0.2), '5': (0.4, 0.5, 0.1, 0.2), '6': (0.5, 0.6, 0.1, 0.2),
            '7': (0.6, 0.7, 0.1, 0.2), '8': (0.7, 0.8, 0.1, 0.2), '9': (0.8, 0.9, 0.1, 0.2),
            '0': (0.9, 1.0, 0.1, 0.2),
            
            # Row 2 (QWERTY)
            'Q': (0, 0.1, 0.2, 0.3), 'W': (0.1, 0.2, 0.2, 0.3), 'E': (0.2, 0.3, 0.2, 0.3),
            'R': (0.3, 0.4, 0.2, 0.3), 'T': (0.4, 0.5, 0.2, 0.3), 'Y': (0.5, 0.6, 0.2, 0.3),
            'U': (0.6, 0.7, 0.2, 0.3), 'I': (0.7, 0.8, 0.2, 0.3), 'O': (0.8, 0.9, 0.2, 0.3),
            'P': (0.9, 1.0, 0.2, 0.3),
            
            # Row 3 (ASDF)
            'A': (0.05, 0.15, 0.3, 0.4), 'S': (0.15, 0.25, 0.3, 0.4), 'D': (0.25, 0.35, 0.3, 0.4),
            'F': (0.35, 0.45, 0.3, 0.4), 'G': (0.45, 0.55, 0.3, 0.4), 'H': (0.55, 0.65, 0.3, 0.4),
            'J': (0.65, 0.75, 0.3, 0.4), 'K': (0.75, 0.85, 0.3, 0.4), 'L': (0.85, 0.95, 0.3, 0.4),
            
            # Row 4 (ZXCV)
            'Z': (0.1, 0.2, 0.4, 0.5), 'X': (0.2, 0.3, 0.4, 0.5), 'C': (0.3, 0.4, 0.4, 0.5),
            'V': (0.4, 0.5, 0.4, 0.5), 'B': (0.5, 0.6, 0.4, 0.5), 'N': (0.6, 0.7, 0.4, 0.5),
            'M': (0.7, 0.8, 0.4, 0.5),
            
            # Row 5 (Special keys)
            'SPACE': (0.2, 0.8, 0.5, 0.6),
            'BACKSPACE': (0.0, 0.2, 0.5, 0.6),
            'ENTER': (0.8, 1.0, 0.5, 0.6)
        }
        
        # Normalize coordinates
        norm_x = x / frame_width
        norm_y = y / frame_height
        
        for key, (x1, x2, y1, y2) in key_regions.items():
            if x1 <= norm_x <= x2 and y1 <= norm_y <= y2:
                return key
        
        return None
    
    def process_key_press(self, key):
        current_time = time.time()
        if current_time - self.last_click_time < self.click_cooldown:
            return
            
        self.last_click_time = current_time
        
        if key == 'SPACE':
            self.typed_text += ' '
        elif key == 'BACKSPACE':
            self.typed_text = self.typed_text[:-1]
        elif key == 'ENTER':
            self.typed_text += '\n'
        else:
            self.typed_text += key
        
        # Update GUI
        self.root.after(0, self.update_text_display)
        
        # Highlight pressed key
        if key in self.key_buttons:
            self.root.after(0, lambda: self.highlight_key(key))
    
    def highlight_key(self, key):
        button = self.key_buttons[key]
        original_style = button.cget('style') if button.cget('style') else 'TButton'
        
        # Create a highlighted style
        style = ttk.Style()
        style.configure('Highlighted.TButton', background='lightgreen')
        
        button.configure(style='Highlighted.TButton')
        self.root.after(200, lambda: button.configure(style=original_style))
    
    def update_text_display(self):
        self.text_display.delete(1.0, tk.END)
        self.text_display.insert(1.0, self.typed_text)
    
    def camera_loop(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hands
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Get index finger tip position
                    index_tip = hand_landmarks.landmark[8]
                    index_x = int(index_tip.x * w)
                    index_y = int(index_tip.y * h)
                    
                    # Draw circle at index finger tip
                    cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), -1)
                    
                    # Check if finger is clicking
                    is_clicking = self.is_finger_clicking(hand_landmarks.landmark)
                    
                    if is_clicking:
                        cv2.circle(frame, (index_x, index_y), 15, (0, 0, 255), 3)
                        
                        # Get key at position
                        key = self.get_key_at_position(index_x, index_y, w, h)
                        if key:
                            self.process_key_press(key)
                            cv2.putText(frame, f'Pressed: {key}', (10, 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Update status
                    status = f"Hand detected - Clicking: {'Yes' if is_clicking else 'No'}"
                    self.root.after(0, lambda s=status: self.update_status(s))
            else:
                self.root.after(0, lambda: self.update_status("No hand detected"))
            
            # Draw keyboard overlay
            self.draw_keyboard_overlay(frame)
            
            # Show frame
            cv2.imshow('Virtual Keyboard Camera', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    def draw_keyboard_overlay(self, frame):
        h, w = frame.shape[:2]
        
        # Draw semi-transparent keyboard overlay
        overlay = frame.copy()
        
        # Define key regions and draw them
        key_regions = {
            # Row 1 (numbers)
            '1': (0, 0.1, 0.1, 0.2), '2': (0.1, 0.2, 0.1, 0.2), '3': (0.2, 0.3, 0.1, 0.2),
            '4': (0.3, 0.4, 0.1, 0.2), '5': (0.4, 0.5, 0.1, 0.2), '6': (0.5, 0.6, 0.1, 0.2),
            '7': (0.6, 0.7, 0.1, 0.2), '8': (0.7, 0.8, 0.1, 0.2), '9': (0.8, 0.9, 0.1, 0.2),
            '0': (0.9, 1.0, 0.1, 0.2),
            
            # Row 2 (QWERTY)
            'Q': (0, 0.1, 0.2, 0.3), 'W': (0.1, 0.2, 0.2, 0.3), 'E': (0.2, 0.3, 0.2, 0.3),
            'R': (0.3, 0.4, 0.2, 0.3), 'T': (0.4, 0.5, 0.2, 0.3), 'Y': (0.5, 0.6, 0.2, 0.3),
            'U': (0.6, 0.7, 0.2, 0.3), 'I': (0.7, 0.8, 0.2, 0.3), 'O': (0.8, 0.9, 0.2, 0.3),
            'P': (0.9, 1.0, 0.2, 0.3),
            
            # Row 3 (ASDF)
            'A': (0.05, 0.15, 0.3, 0.4), 'S': (0.15, 0.25, 0.3, 0.4), 'D': (0.25, 0.35, 0.3, 0.4),
            'F': (0.35, 0.45, 0.3, 0.4), 'G': (0.45, 0.55, 0.3, 0.4), 'H': (0.55, 0.65, 0.3, 0.4),
            'J': (0.65, 0.75, 0.3, 0.4), 'K': (0.75, 0.85, 0.3, 0.4), 'L': (0.85, 0.95, 0.3, 0.4),
            
            # Row 4 (ZXCV)
            'Z': (0.1, 0.2, 0.4, 0.5), 'X': (0.2, 0.3, 0.4, 0.5), 'C': (0.3, 0.4, 0.4, 0.5),
            'V': (0.4, 0.5, 0.4, 0.5), 'B': (0.5, 0.6, 0.4, 0.5), 'N': (0.6, 0.7, 0.4, 0.5),
            'M': (0.7, 0.8, 0.4, 0.5),
            
            # Row 5 (Special keys)
            'SPACE': (0.2, 0.8, 0.5, 0.6),
            'BACKSPACE': (0.0, 0.2, 0.5, 0.6),
            'ENTER': (0.8, 1.0, 0.5, 0.6)
        }
        
        for key, (x1, x2, y1, y2) in key_regions.items():
            x1_px, x2_px = int(x1 * w), int(x2 * w)
            y1_px, y2_px = int(y1 * h), int(y2 * h)
            
            # Draw rectangle
            cv2.rectangle(overlay, (x1_px, y1_px), (x2_px, y2_px), (255, 255, 255), -1)
            cv2.rectangle(overlay, (x1_px, y1_px), (x2_px, y2_px), (0, 0, 0), 2)
            
            # Draw key label
            text_x = x1_px + (x2_px - x1_px) // 2
            text_y = y1_px + (y2_px - y1_px) // 2
            
            text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x -= text_size[0] // 2
            text_y += text_size[1] // 2
            
            cv2.putText(overlay, key, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Blend overlay with original frame
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    def update_status(self, status):
        self.status_label.config(text=f"Status: {status}")
    
    def quit_app(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.quit()
    
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)
        self.root.mainloop()

# Installation requirements (run these commands in terminal):
"""
pip install opencv-python
pip install mediapipe
pip install numpy
"""

if __name__ == "__main__":
    print("Starting Virtual Hand-Controlled Keyboard...")
    print("Make sure you have installed the required packages:")
    print("pip install opencv-python mediapipe numpy")
    print("\nInstructions:")
    print("1. Point your index finger at the camera")
    print("2. Move your finger to hover over keys")
    print("3. Bring your thumb close to your index finger to 'click'")
    print("4. Press 'q' in the camera window to quit")
    
    app = VirtualKeyboard()
    app.run()