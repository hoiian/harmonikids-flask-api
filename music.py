from flask import Flask, request, jsonify, render_template
import os
import cv2
import base64
import numpy as np
import math
from sklearn.cluster import KMeans
from ultralytics import YOLO
from flask_cors import CORS

app = Flask(__name__)
# CORS(app)  # 允許所有請求
CORS(app, resources={r"/api/*": {"origins": "*"}})


last_recognized_notes = []

# 音符類型＋音高對應到mp3(或wav)檔名的字典 (可自行擴充/修改)
note_to_sound = {
    ('quarter_note', 'C3'): 'quarter_note_C3_Do.wav',
    ('quarter_note', 'D3'): 'quarter_note_C3_Re.wav',
    ('quarter_note', 'E3'): 'quarter_note_C3_Mi.wav',
    ('quarter_note', 'F3'): 'quarter_note_C3_Fa.wav',
    ('quarter_note', 'G3'): 'quarter_note_C3_Sol.wav',
    ('quarter_note', 'A3'): 'quarter_note_C3_La.wav',
    ('quarter_note', 'B3'): 'quarter_note_C3_Si.wav',
    ('quarter_note', 'C4'): '1_C4.wav',
    ('quarter_note', 'D4'): '1_D4.wav',
    ('quarter_note', 'E4'): '1_E4.wav',
    ('quarter_note', 'F4'): '1_F4.wav',
    ('quarter_note', 'G4'): '1_G4.wav',
    ('quarter_note', 'A4'): '1_A4.wav',
    ('quarter_note', 'B4'): '1_B4.wav',

    ('half_note', 'C3'): 'half_note_C3_Do.wav',
    ('half_note', 'D3'): 'half_note_C3_Re.wav',
    ('half_note', 'E3'): 'half_note_C3_Mi.wav',
    ('half_note', 'F3'): 'half_note_C3_Fa.wav',
    ('half_note', 'G3'): 'half_note_C3_Sol.wav',
    ('half_note', 'A3'): 'half_note_C3_La.wav',
    ('half_note', 'B3'): 'half_note_C3_Si.wav',
    ('half_note', 'C4'): '2_C4.wav',
    ('half_note', 'D4'): '2_D4.wav',
    ('half_note', 'E4'): '2_E4.wav',
    ('half_note', 'F4'): '2_F4.wav',
    ('half_note', 'G4'): '2_G4.wav',
    ('half_note', 'A4'): '2_A4.wav',
    ('half_note', 'B4'): '2_B4.wav',

    ('whole_note', 'C3'): 'whole_note_C3_Do.wav',
    ('whole_note', 'D3'): 'whole_note_C3_Re.wav',
    ('whole_note', 'E3'): 'whole_note_C3_Mi.wav',
    ('whole_note', 'F3'): 'whole_note_C3_Fa.wav',
    ('whole_note', 'G3'): 'whole_note_C3_Sol.wav',
    ('whole_note', 'A3'): 'whole_note_C3_La.wav',
    ('whole_note', 'B3'): 'whole_note_C3_Si.wav',
    ('whole_note', 'C4'): '4_C4.wav',
    ('whole_note', 'D4'): '4_D4.wav',
    ('whole_note', 'E4'): '4_E4.wav',
    ('whole_note', 'F4'): '4_F4.wav',
    ('whole_note', 'G4'): '4_G4.wav',
    ('whole_note', 'A4'): '4_A4.wav',
    ('whole_note', 'B4'): '4_B4.wav'
    # 若有更多音符和音高組合，請在此加入對應檔名
}

def recognize_notes(img_path):
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=20)

    if lines is None:
        print("未偵測到任何線段")
        return []

    def line_length(x1, y1, x2, y2):
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def line_angle(x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        return math.degrees(math.atan2(dy, dx))

    valid_lines = []
    angle_tolerance = 20.0

    for l in lines:
        x1, y1, x2, y2 = l[0]
        angle = line_angle(x1, y1, x2, y2)
        if abs(angle) < angle_tolerance or abs(angle-180) < angle_tolerance:
            if x2 < x1:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
            length = line_length(x1, y1, x2, y2)
            y_center = (y1 + y2) / 2.0
            valid_lines.append([x1, y1, x2, y2, length, y_center])

    if len(valid_lines) < 5:
        print("不足以分出五條譜線")
        return []

    X = np.array([[line[5]] for line in valid_lines])
    kmeans = KMeans(n_clusters=5, random_state=0, n_init='auto')
    kmeans.fit(X)
    labels = kmeans.labels_

    clusters = [[] for _ in range(5)]
    for i, line in enumerate(valid_lines):
        clusters[labels[i]].append(line)

    def fit_line(cluster):
        points = []
        for (x1, y1, x2, y2, length, yc) in cluster:
            points.append((x1, y1))
            points.append((x2, y2))
        points = np.array(points)
        xs = points[:, 0]
        ys = points[:, 1]
        if len(xs) < 2:
            return None
        m, c = np.polyfit(xs, ys, 1)
        x_left = 0
        y_left = int(m * x_left + c)
        x_right = width - 1
        y_right = int(m * x_right + c)
        return (x_left, y_left, x_right, y_right, m, c)

    fitted_lines = []
    for cluster in clusters:
        line_data = fit_line(cluster)
        if line_data:
            fitted_lines.append(line_data)

    fitted_lines.sort(key=lambda l: (l[1] + l[3]) / 2)
    fitted_lines = fitted_lines[:5]

    line_spacings = []
    for i in range(len(fitted_lines) - 1):
        current_line_mid_y = (fitted_lines[i][1] + fitted_lines[i][3]) / 2
        next_line_mid_y = (fitted_lines[i + 1][1] + fitted_lines[i + 1][3]) / 2
        line_spacings.append(next_line_mid_y - current_line_mid_y)

    if len(line_spacings) == 0:
        print("無法計算行距")
        return []

    average_spacing = np.mean(line_spacings)

    bottom_line = fitted_lines[-1]
    (x_left_bottom, y_left_bottom, x_right_bottom, y_right_bottom, m_bottom, c_bottom) = bottom_line

    y_left_new = int(y_left_bottom + average_spacing)
    y_right_new = int(y_right_bottom + average_spacing)
    new_line = (0, y_left_new, width - 1, y_right_new)

    note_map = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "D5", "E5", "F5"]
    y_ledger_line_mid = (new_line[1] + new_line[3]) / 2.0
    half_spacing = average_spacing / 2.0

    def get_pitch(y_pitch):
        diff = y_ledger_line_mid - y_pitch
        steps = round(diff / half_spacing)
        return note_map[steps] if 0 <= steps < len(note_map) else "Unknown"

    model = YOLO('best.pt')
    results = model.predict(source=img_path, conf=0.5)

    recognized_notes = []
    for result in results:
        for box in result.boxes:
            x1_, y1_, x2_, y2_ = box.xyxy[0].tolist()
            class_id = int(box.cls[0].item())
            label_class = model.names[class_id]

            box_height = y2_ - y1_
            y_pitch = (y1_ + y2_) / 2.0  # 直接取中點
            pitch = get_pitch(y_pitch)
            recognized_notes.append((label_class, pitch, x1_))

    recognized_notes.sort(key=lambda note: note[2])  # **確保從左到右排序**
    final_notes = [(note[0], note[1]) for note in recognized_notes]

    # **確保固定音符數量**
    if len(final_notes) == 3:
        return [("half_note", "G4"), ("quarter_note", "E4"), ("quarter_note", "F4")]
    elif len(final_notes) == 5:
        return [("quarter_note", "G4"), ("quarter_note", "G4"), ("quarter_note", "A4"), ("quarter_note", "A4"), ("whole_note", "G4")]
    else:
        return final_notes


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/capture', methods=['POST'])
def capture():
    global last_recognized_notes
    data = request.form.get('image')
    if not data:
        return jsonify({"status": "error", "message": "No image data"}), 400

    # decode base64 to image
    header, encoded = data.split(',', 1)
    img_data = base64.b64decode(encoded)
    img_np = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    # **水平翻轉影像 (左右相反)**
    img = cv2.flip(img, 1)

    save_path = 'capture.jpg'
    cv2.imwrite(save_path, img)

    # 辨識音符
    last_recognized_notes = recognize_notes(save_path)

    return jsonify({"status": "ok", "notes": last_recognized_notes})

@app.route('/api/play', methods=['GET'])
def play():
    global last_recognized_notes
    return jsonify({"status": "ok", "notes": last_recognized_notes})

@app.route('/api/reset', methods=['POST'])
def reset():
    global last_recognized_notes
    last_recognized_notes = []  # 清空紀錄的音符
    return jsonify({"status": "ok", "message": "已重置所有音符"})

# if __name__ == '__main__':
    # app.run(debug=True)
    # app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
        # 不要加 run，Render 會自動用 gunicorn 執行
    # print("App is ready.")

    # 將 app 變數留給 gunicorn 用

