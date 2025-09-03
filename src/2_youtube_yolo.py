from ultralytics import YOLO

model = YOLO("C:\Potenup\DeepLearning-YOLO-Study\models\yolo11n.pt") # 모델 저장 경로 작성

youtube_url = "https://youtu.be/30RTZ6zbM5g"

results = model(youtube_url, stream=True)

for res in results:
    print(res.boxes.cls)