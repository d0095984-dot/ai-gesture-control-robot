import cv2
import numpy as np
from IPython.display import display, Javascript, Image
from google.colab.output import eval_js
from base64 import b64decode
import time




# ─── Camera Function ───────────────────────────────
def start_camera():
    display(Javascript('''
        window._stream = null;
        window._video = null;
        async function startCam() {
            window._stream = await navigator.mediaDevices.getUserMedia({video: true});
            window._video = document.createElement('video');
            window._video.srcObject = window._stream;
            await window._video.play();
        }
        startCam();
    '''))
    time.sleep(2)


def grab_frame():
    data = eval_js('''
        (function() {
            if (!window._video || window._video.readyState < 2) return "";
            const canvas = document.createElement('canvas');
            canvas.width = window._video.videoWidth;
            canvas.height = window._video.videoHeight;
            canvas.getContext('2d').drawImage(window._video, 0, 0);
            return canvas.toDataURL('image/jpeg', 0.75);
        })()
    ''')
    if not data or ',' not in data:
        return None
    binary = b64decode(data.split(',')[1])
    arr = np.frombuffer(binary, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def stop_camera():
    display(Javascript('''
        if (window._stream) {
            window._stream.getTracks().forEach(t => t.stop());
        }
    '''))




# ─── Gesture Helpers ───────────────────────────────
def get_defects(contour):
    hull_idx = cv2.convexHull(contour, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 3:
        return None
    try:
        return cv2.convexityDefects(contour, hull_idx)
    except:
        return None


def count_defect_gaps(defects, min_depth=18.0):
    if defects is None:
        return 0
    return sum(1 for i in range(defects.shape[0]) if defects[i, 0, 3] / 256.0 > min_depth)




def classify_gesture(contour):
    area = cv2.contourArea(contour)
    if area < 7000:
        return "No Hand", (160, 160, 160)
    
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = area / hull_area if hull_area > 0 else 0
    _, _, w, h = cv2.boundingRect(contour)
    ar = w / h if h > 0 else 1.0
    gaps = count_defect_gaps(get_defects(contour))


    # Корейское сердечко
    if solidity > 0.75 and gaps <= 2 and 0.55 < ar < 1.45 and area < 35000:
        return "ASIAN HEART", (255, 105, 180)
    
    # OK
    if 1.5 <= gaps <= 3.5 and 0.65 < solidity < 0.88 and ar < 1.3:
        return "OK", (0, 255, 200)
    
    # Кулак
    if ar > 1.3 or solidity < 0.73:
        return "FORWARD", (0, 255, 0)
    
    # Открытая ладонь
    return "STOP", (0, 60, 255)




# ─── Drawing ───────────────────────────────
def draw_robot(frame):
    h, fw = frame.shape[:2]
    cx = fw - 80
    cv2.rectangle(frame, (cx-40, h-160), (cx+40, h-60), (70, 70, 70), -1)
    cv2.circle(frame, (cx, h-185), 28, (0, 150, 255), -1)
    cv2.circle(frame, (cx-10, h-188), 5, (255, 255, 255), -1)
    cv2.circle(frame, (cx+10, h-188), 5, (255, 255, 255), -1)


def draw_hud(frame, gesture, color):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 95), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, f"Gesture: {gesture}", (30, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv2.LINE_AA)




# ─── Main Loop ───────────────────────────────
print("🤖 Gesture Control запущен")
print("Жесты: ASIAN HEART ❤️ | OK 👌 | FORWARD ✊ | STOP ✋")


start_camera()


for i in range(180):
    frame = grab_frame()
    if frame is None:
        time.sleep(0.1)
        continue


    frame = cv2.flip(frame, 1)


    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 20, 60]), np.array([20, 255, 255]))
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    gesture_label = "No Hand"
    label_color = (160, 160, 160)


    if contours:
        best = max(contours, key=cv2.contourArea)
        if cv2.contourArea(best) > 7000:
            x, y, w, h = cv2.boundingRect(best)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            gesture_label, label_color = classify_gesture(best)


    draw_hud(frame, gesture_label, label_color)
    draw_robot(frame)


    display(Image(data=cv2.imencode('.jpg', frame)[1].tobytes()))
    time.sleep(0.12)


stop_camera()
print("✅ Демо завершено!")


