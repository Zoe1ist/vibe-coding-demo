import cv2
from transformers import pipeline

print("正在加载 AI 模型，请稍等...")

emotion_model = pipeline(
    "image-classification",
    model="dima806/facial_emotions_image_detection"
)

print("模型加载完成！")
print("按 Q 键退出。")

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()

    if not ret:
        print("无法读取摄像头")
        break

    # OpenCV 使用 BGR，模型更适合 RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # AI 进行情绪分类
    result = emotion_model(rgb_frame)

    emotion = result[0]["label"]
    confidence = result[0]["score"]

    # 在摄像头画面上显示结果
    text = f"{emotion}: {confidence:.1%}"

    cv2.putText(
        frame,
        text,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("AI Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
