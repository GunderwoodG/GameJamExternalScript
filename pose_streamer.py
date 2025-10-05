#!/usr/bin/env python3
"""
Optimized MediaPipe Pose Streamer for Godot
Sends pose landmark data over TCP with performance optimizations
"""

import cv2
import mediapipe as mp
import socket
import json
import time
import sys

# Configuration
HOST = '127.0.0.1'
PORT = 5050
TARGET_FPS = 30  # Limit to 30 FPS to reduce load
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

class PoseStreamer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # Use lighter model (0=lite, 1=full, 2=heavy)
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.server = None
        self.client = None
        self.cap = None
        self.running = True
        
    def setup_server(self):
        """Initialize TCP server"""
        try:
            self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server.bind((HOST, PORT))
            self.server.listen(1)
            self.server.settimeout(60)
            print(f"🎯 Server listening on {HOST}:{PORT}")
            return True
        except Exception as e:
            print(f"❌ Server setup failed: {e}")
            return False
    
    def wait_for_client(self):
        """Wait for Godot to connect"""
        print("⏳ Waiting for Godot client...")
        print("   (Start Godot now if you haven't already)")
        try:
            self.client, addr = self.server.accept()
            self.client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.client.setblocking(True)  # Ensure blocking mode
            print(f"✅ Client connected: {addr}")
            print("   Connection established successfully!")
            time.sleep(0.5)  # Brief pause for connection stability
            return True
        except socket.timeout:
            print("❌ Connection timeout - Godot didn't connect in 60 seconds")
            return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def setup_camera(self):
        """Initialize camera with optimized settings"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ Cannot open camera")
            return False
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag
        
        print(f"🎥 Camera initialized: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {TARGET_FPS}fps")
        return True
    
    def extract_landmarks(self, results):
        """Extract pose data efficiently"""
        if not results.pose_landmarks:
            return {}
        
        landmarks = {}
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            # Use index-based naming for efficiency
            name = self.mp_pose.PoseLandmark(idx).name
            landmarks[name] = {
                "x": round(landmark.x, 3),
                "y": round(landmark.y, 3),
                "z": round(landmark.z, 3),
                "v": round(landmark.visibility, 3)  # Shortened key
            }
        return landmarks
    
    def send_data(self, data):
        """Send JSON data to client"""
        try:
            message = json.dumps(data, separators=(',', ':')) + "\n"  # Compact JSON
            self.client.sendall(message.encode('utf-8'))
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            return False
    
    def run(self):
        """Main streaming loop"""
        if not self.setup_server():
            return
        
        if not self.wait_for_client():
            self.cleanup()
            return
        
        if not self.setup_camera():
            self.cleanup()
            return
        
        print("\n🚀 Streaming started! Press 'q' or ESC to quit.\n")
        
        frame_time = 1.0 / TARGET_FPS
        frame_count = 0
        fps_start = time.time()
        last_fps_print = time.time()
        
        try:
            while self.running:
                loop_start = time.time()
                
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠️ Frame capture failed")
                    break
                
                # Process with MediaPipe (skip RGB conversion overhead when not needed)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False  # Performance optimization
                results = self.pose.process(frame_rgb)
                frame_rgb.flags.writeable = True
                
                # Extract and send landmarks
                landmarks = self.extract_landmarks(results)
                if not self.send_data(landmarks):
                    print("❌ Client disconnected")
                    break
                
                # Draw landmarks on preview (optional, can be disabled for performance)
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
                
                # Display FPS on frame
                current_time = time.time()
                frame_count += 1
                if current_time - fps_start >= 1.0:
                    fps = frame_count / (current_time - fps_start)
                    frame_count = 0
                    fps_start = current_time
                    
                    # Print FPS every 2 seconds
                    if current_time - last_fps_print >= 2.0:
                        print(f"📊 FPS: {fps:.1f} | Landmarks: {len(landmarks)}")
                        last_fps_print = current_time
                    
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show preview
                cv2.imshow("Pose Streamer", frame)
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    print("\n👋 Quit requested")
                    break
                
                # Frame rate limiting
                elapsed = time.time() - loop_start
                sleep_time = frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("\n🧹 Cleaning up...")
        self.running = False
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        if self.client:
            self.client.close()
        if self.server:
            self.server.close()
        self.pose.close()
        
        print("✅ Shutdown complete")

def main():
    print("=" * 50)
    print("MediaPipe Pose Streamer for Godot")
    print("=" * 50)
    
    streamer = PoseStreamer()
    streamer.run()

if __name__ == "__main__":
    main()

#v2# # pose_streamer.py
# import cv2
# import mediapipe as mp
# import socket
# import json
# import time

# # Setup MediaPipe Pose
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(
#     static_image_mode=False,
#     model_complexity=1,
#     enable_segmentation=False,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# # TCP Server Configuration
# HOST = '127.0.0.1'
# PORT = 5050

# def create_server():
#     """Create and configure the TCP server socket"""
#     server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#     server.bind((HOST, PORT))
#     server.listen(1)
#     server.settimeout(30)  # 30 second timeout for connection
#     return server

# def wait_for_client(server):
#     """Wait for Godot client to connect"""
#     print(f"🎯 Waiting for Godot to connect on {HOST}:{PORT}...")
#     try:
#         client, address = server.accept()
#         client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # Disable Nagle's algorithm
#         client.settimeout(None)  # No timeout for send operations
#         print(f"✅ Godot connected from {address}")
#         # Give Godot a moment to fully establish connection
#         time.sleep(0.5)
#         return client
#     except socket.timeout:
#         print("❌ Connection timeout - no client connected")
#         return None

# def extract_pose_data(results):
#     """Extract pose landmarks into a clean dictionary"""
#     keypoints = {}
    
#     if results.pose_landmarks:
#         for lm in mp_pose.PoseLandmark:
#             landmark = results.pose_landmarks.landmark[lm]
#             keypoints[lm.name] = {
#                 "x": round(landmark.x, 3),
#                 "y": round(landmark.y, 3),
#                 "z": round(landmark.z, 3),
#                 "visibility": round(landmark.visibility, 3)
#             }
    
#     return keypoints

# def main():
#     server = None
#     client = None
#     cap = None
    
#     try:
#         # Setup server
#         server = create_server()
#         client = wait_for_client(server)
        
#         if client is None:
#             return
        
#         # Start webcam
#         cap = cv2.VideoCapture(0)
#         if not cap.isOpened():
#             print("❌ Could not open webcam")
#             return
        
#         print("🎥 Webcam started. Press ESC to quit.")
        
#         frame_count = 0
#         fps_time = time.time()
        
#         while cap.isOpened():
#             success, frame = cap.read()
#             if not success:
#                 print("⚠️ Failed to read frame")
#                 break
            
#             # Process frame with MediaPipe
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = pose.process(frame_rgb)
            
#             # Extract pose data
#             keypoints = extract_pose_data(results)
            
#             # Send data to Godot
#             try:
#                 message = json.dumps(keypoints) + "\n"
#                 client.sendall(message.encode('utf-8'))
#             except (BrokenPipeError, ConnectionResetError, OSError) as e:
#                 print(f"❌ Connection lost: {e}")
#                 break
#             except Exception as e:
#                 print(f"❌ Unexpected error: {e}")
#                 break
            
#             # Display FPS
#             frame_count += 1
#             if frame_count % 30 == 0:
#                 current_time = time.time()
#                 fps = 30 / (current_time - fps_time)
#                 fps_time = current_time
#                 print(f"📊 FPS: {fps:.1f} | Landmarks: {len(keypoints)}")
            
#             # Show preview window
#             if results.pose_landmarks:
#                 # Draw landmarks on frame (optional)
#                 mp.solutions.drawing_utils.draw_landmarks(
#                     frame,
#                     results.pose_landmarks,
#                     mp_pose.POSE_CONNECTIONS
#                 )
            
#             cv2.imshow("Pose Tracker", frame)
            
#             # ESC to quit
#             if cv2.waitKey(5) & 0xFF == 27:
#                 print("👋 ESC pressed, shutting down...")
#                 break
    
#     except KeyboardInterrupt:
#         print("\n👋 Keyboard interrupt, shutting down...")
    
#     except Exception as e:
#         print(f"❌ Fatal error: {e}")
    
#     finally:
#         # Cleanup
#         print("🧹 Cleaning up...")
#         if cap is not None:
#             cap.release()
#         cv2.destroyAllWindows()
#         if client is not None:
#             client.close()
#         if server is not None:
#             server.close()
#         pose.close()
#         print("✅ Cleanup complete")

# if __name__ == "__main__":
#     main()

# v1
# # pose_streamer.py

# import cv2
# import mediapipe as mp
# import socket
# import json

# # Setup MediaPipe
# mp_pose = mp.solutions.pose
# # pose = mp_pose.Pose()
# pose = mp_pose.Pose(static_image_mode=False,
#                     model_complexity=1,
#                     enable_segmentation=False,
#                     min_detection_confidence=0.5,
#                     min_tracking_confidence=0.5)

# # Setup TCP server
# HOST = '127.0.0.1'
# PORT = 5050
# server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server.bind((HOST, PORT))
# server.listen(1)
# print("Waiting for Godot to connect...")
# client, _ = server.accept()
# print("Godot connected!")

# # Start webcam
# cap = cv2.VideoCapture(0)



# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         break

#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = pose.process(frame_rgb)

#     keypoints = {}

#     if results.pose_landmarks:
#         for lm in mp_pose.PoseLandmark:
#             landmark = results.pose_landmarks.landmark[lm]
#             keypoints[lm.name] = {
#                 "x": round(landmark.x, 3),
#                 "y": round(landmark.y, 3),
#                 "z": round(landmark.z, 3),
#                 "visibility": round(landmark.visibility, 3)
#             }
#     else:
#         # Send empty pose to keep stream valid
#         keypoints["RIGHT_WRIST"] = {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.0}
#         keypoints["RIGHT_SHOULDER"] = {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.0}
#         keypoints["LEFT_SHOULDER"] = {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.0}

#     try:
#         client.sendall((json.dumps(keypoints) + "\n").encode("utf-8"))
#     except:
#         print("Connection lost.")
#         break

#     cv2.imshow("Pose Tracker", frame)
#     if cv2.waitKey(5) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()
# client.close()
# server.close()