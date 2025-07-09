import cv2
import mediapipe as mp

# khơi tạo các đối tượng MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0) 
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("empty camera frame.")
        continue

    # Chuyển đổi khung hình sang định dạng RGB để xử lý
    frame.flags.writeable = False
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # xử lý khung hình 
    results = pose.process(rgb_frame)
    frame.flags.writeable = True
    # Hiện thị kết quả lên frame
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

    # Hiện thị kết quả lên cửa sổ
    cv2.imshow('MediaPipe Pose Detection', cv2.flip(frame, 1))

    # Exit
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
pose.close()