# 🎵 HarmoniKids - 音符辨識遊戲

這是一個基於 **Flask（Python）+ React（JavaScript）** 的音符辨識遊戲。  
通過 **YOLO 物件傾機學習** 來辨識五線譜上的音符，并將結果顯示在 React 前端應用中。

---

## **🚀 功能**

- 📸 **相機擁採**：通過攝影機拍攝樂譜
- 🧠 **YOLO 音符辨識**：使用機器學習辨識音符
- 🎶 **音符播放**：辨識後可播放對應音符
- 🎮 **遊戲模式**：React 介面與 Flask 互動

---

## **👤 專案結構**

```
HarmoniKids-Dev/
│️─ game/                  # 🎮 React 前端 (UI)
│   └️ src/               # React 應用核心代碼
│   └️ public/            # 靜態資源
│   └️ package.json       # React 依賴 & 設定
│   └️ .gitignore         # 忽略 node_modules
│️─ templates/             # Flask 的 HTML 檔案
│️─ static/                # Flask 靜態資源 (JS, CSS, images)
│️─ myenv/                 # Python 虛擬環境 (GitHub 忽略)
│️─ music.py               # 🎵 Flask 後端主程序
│️─ requirements.txt       # 📞 Flask 依賴
│️─ .gitignore             # 🔥 忽略 myenv & node_modules
│️─ README.md              # 📚 此檔案
```

---

## **💡 環境安裝**

### **1️⃣ 安裝 Flask (後端)**

```sh
# 進入專案資料夾
cd HarmoniKids-Dev

# 建立 Python 虛擬環境
python3 -m venv myenv

# 啟動虛擬環境 (Mac/Linux)
source myenv/bin/activate

# 安裝 Flask 及相關依賴
pip install -r requirements.txt
```

### **2️⃣ 安裝 React (前端)**

```sh
# 進入 React 資料夾
cd game

# 安裝 React 依賴
npm install
```

---

## **🚀 啟動專案**

### **✅ 1. 啟動 Flask (後端)**

```sh
# 進入 Flask 專案
cd HarmoniKids-Dev

# 啟動虛擬環境
source myenv/bin/activate

# 啟動 Flask 伺服器
python3 music.py
```

📌 Flask 預設運行在 `http://127.0.0.1:5000/`

---

### **✅ 2. 啟動 React (前端)**

```sh
# 進入 React 專案
cd game

# 啟動 React 開發伺服器
npm start
```

📌 React 會運行在 `http://localhost:3000/`

---

## **🌍 API 端點**

| **HTTP 方法** | **端點**       | **說明**                 |
| ------------- | -------------- | ------------------------ |
| `GET`         | `/api/play`    | 取得辨識後的音符         |
| `POST`        | `/api/capture` | 上傳相機拍攝的樂譜並辨識 |
| `POST`        | `/api/play`    | 播放音符 (未來支援)      |

---
