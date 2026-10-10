# 🚗 Vehicle Speed Detection System Using YOLO & Streamlit

**Real-Time Vehicle Detection, Tracking, and Speed Estimation Using a Webcam**

**Designed by Engr. Bilal Mehmood**

## 📌 Project Overview

The Vehicle Speed Detection System is a Python-based computer vision application developed using **Streamlit, Ultralytics YOLO, OpenCV, and ByteTrack**.

The application uses a webcam to detect and track vehicles in real time, including cars, motorcycles, buses, and trucks. It estimates vehicle speed by measuring the time taken to cross two reference lines separated by a known physical distance.

The system provides a live video feed, vehicle tracking information, estimated speed in km/h, and downloadable speed measurement history.

## ✨ Key Features

* 📹 Real-time webcam video processing
* 🚗 Car detection
* 🏍️ Motorcycle detection
* 🚌 Bus detection
* 🚚 Truck detection
* 🎯 YOLO-based object detection
* 🆔 Vehicle tracking using ByteTrack
* 📏 Configurable distance between measurement lines
* ⏱️ Speed estimation in kilometers per hour (km/h)
* 📊 Live vehicle information dashboard
* 🧾 Speed measurement history
* 📥 CSV export functionality
* ⚙️ Adjustable detection confidence
* 🌐 Streamlit web interface
* ☁️ GitHub and Streamlit Community Cloud deployment support

## 🛠️ Technologies Used

| Technology       | Purpose                     |
| ---------------- | --------------------------- |
| Python           | Core programming language   |
| Streamlit        | Web application interface   |
| Ultralytics YOLO | Vehicle detection           |
| ByteTrack        | Object tracking             |
| OpenCV           | Image and video processing  |
| Streamlit-WebRTC | Browser webcam streaming    |
| NumPy            | Numerical operations        |
| Pandas           | Tabular data and CSV export |
| PyAV             | Video frame processing      |

## 📁 Project Structure

```text
VehicleSpeedDetection/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 💻 System Requirements

* Python 3.11 or 3.12 recommended
* Webcam or compatible USB camera
* Internet connection for initial model download
* Modern web browser
* Sufficient CPU or GPU resources for real-time processing

## 📦 Installation

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd VehicleSpeedDetection
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual GitHub repository URL.

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Launch the Application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

Allow webcam access when prompted by your browser.

## 🚦 How Vehicle Speed Estimation Works

The application uses two horizontal reference lines to estimate a vehicle's speed.

1. The webcam captures video frames.
2. YOLO detects supported vehicle classes.
3. ByteTrack assigns tracking IDs to detected vehicles.
4. The system records when a tracked vehicle crosses each reference line.
5. The elapsed time is calculated.
6. The speed is estimated using the known distance between the lines.

### Speed Calculation Formula

**Speed (km/h) = Distance (meters) ÷ Time (seconds) × 3.6**

For example, if a vehicle travels 5 meters in 0.5 seconds:

Speed = (5 ÷ 0.5) × 3.6 = **36 km/h**

This is an illustrative example, not a measured result from the application.

## ⚙️ Configuration

The application provides configurable settings in the sidebar:

* **YOLO model:** Select a supported model, such as `yolo11n.pt`, `yolov8n.pt`, or `yolo11s.pt`.
* **Detection confidence:** Adjust the confidence threshold.
* **Distance between lines:** Enter the actual measured distance in meters.
* **Line A position:** Adjust the first reference line.
* **Line B position:** Adjust the second reference line.

Ensure that Line B is below Line A in the displayed image.

For meaningful measurements, use a fixed camera and position the reference lines appropriately along the vehicle's direction of travel.

## ☁️ Streamlit Community Cloud Deployment

### Step 1: Create a GitHub Repository

Create a new repository on GitHub named:

`VehicleSpeedDetection`

### Step 2: Upload Project Files

Upload these files to the repository root:

* `app.py`
* `requirements.txt`
* `README.md`

You may also include a `.gitignore` file.

### Step 3: Deploy on Streamlit Community Cloud

1. Visit https://share.streamlit.io/
2. Sign in with GitHub.
3. Select **Create app**.
4. Choose your repository and branch.
5. Set the main file path to `app.py`.
6. Click **Deploy**.

### Step 4: Configure Webcam Access

The application uses browser-based WebRTC streaming. Webcam permissions, HTTPS, and network connectivity must be configured correctly.

Depending on the hosting environment, additional STUN/TURN or firewall configuration may be necessary.

**Important:** A successful deployment does not guarantee that webcam streaming will work on every network.

## 📊 Dashboard Output

The application displays:

* Number of currently tracked vehicles
* Video processing FPS
* Number of tracked vehicles with speed estimates
* Vehicle tracking IDs
* Vehicle classes and detection confidence
* Estimated vehicle speed in km/h
* Speed measurement history
* Downloadable CSV records

## ⚠️ Limitations and Accuracy

This project is designed for educational, research, and experimental traffic-monitoring applications.

Speed estimates may be affected by:

* Camera angle and perspective
* Incorrect distance calibration
* Occlusion and tracking ID changes
* Dropped or delayed video frames
* Low lighting and motion blur
* Vehicles changing lanes or direction
* Webcam frame-rate variability

The prototype uses processing timestamps, which may not correspond exactly to the original video capture timestamps. For improved accuracy, use reliable frame timestamps, calibrated road geometry, consistent tracking, and validation against a known speed reference.

**Do not use unvalidated estimates for legal enforcement or safety-critical decisions.**

## 🔒 Privacy and Responsible Use

* Obtain appropriate permission before monitoring people or vehicles.
* Comply with applicable privacy and data protection requirements.
* Avoid collecting personally identifiable information unnecessarily.
* Secure any stored traffic footage or records.
* Inform users when video monitoring is active.

## 🔮 Future Enhancements

Possible future features include:

* Overspeeding alerts and configurable speed limits
* Red bounding boxes for vehicles exceeding the speed threshold
* Vehicle counting and traffic-density analysis
* License plate recognition, where lawful and appropriate
* Vehicle classification and direction detection
* Annotated video recording
* PDF and Excel report generation
* Database integration
* Improved perspective calibration and speed validation
* Multi-camera traffic monitoring

## 👨‍💻 Author

**Engr. Bilal Mehmood**
AI Course Director
PITAC Regional Center, Karachi, Sindh

## 📄 License

This project is intended for educational and research purposes. Add an appropriate open-source license file to your repository if you intend to distribute the project.

---

**Built with Python, Streamlit, OpenCV, and Ultralytics YOLO.**
