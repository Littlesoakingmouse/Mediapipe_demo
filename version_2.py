import cv2
import mediapipe as mp
# khổi tạo các đối tượng MediaPipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2 
latest_result = None

# nhận và lưu kết quả từ callback
def save_result(result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result
# khởi tạo các tùy chọn cho PoseLandmarker(thư viện)
options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='pose_landmarker_lite.task'),
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=save_result)
# khởi tạo PoseLandmarker 
with vision.PoseLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Chuyển đổi khung hình sang định dạng RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # Lấy thời gian hiện tại để sử dụng trong callback
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        
        # gọi hàm detect_async để phát hiện các điểm mốc trên cơ thể
        # và lưu kết quả vào latest_result
        landmarker.detect_async(mp_image, timestamp_ms)

        # Tạo một bản sao của khung hình để vẽ các điểm mốc
        annotated_image = frame.copy()

        if latest_result and latest_result.pose_landmarks:
            # chạy qua từng danh sách các điểm 
            # trong kết quả của PoseLandmarker
            for pose_landmarks_list in latest_result.pose_landmarks:
                # Chuyển đổi các điểm mốc thành định dạng protobuf (idk)
                # để vẽ lên khung hình
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                pose_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks_list
                ])

                # vẽ các điểm mốc lên khung hình
                mp.solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    pose_landmarks_proto,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_pose_landmarks_style())
        
        # Hiện thị kết quả lên cửa sổ
        cv2.imshow('MediaPipe Pose Landmarker', cv2.flip(annotated_image, 1))
        # Exit = Esc
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()