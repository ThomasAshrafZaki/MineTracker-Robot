
#  Autonomous Mine-Clearing Robot 
### *Empowering Demining Teams with AI & Robotics*

![Project Status](https://img.shields.io/badge/Status-Active_Development-green)
![Tech Stack](https://img.shields.io/badge/Stack-C%2B%2B%20%7C%20Python%20%7C%20YOLOv8-blue)

##  Overview
This project is an integrated autonomous system designed to assist demining technicians (like our persona **Tariq**) in hazardous environments. It combines high-precision motor control with advanced computer vision to detect metallic objects and navigate complex terrains safely.

---

##  Features

###  Brain (Vision System - Python)
* **YOLOv8 Detection:** Real-time object detection with adaptive confidence based on lighting conditions.
* **Dynamic ROI:** Uses Optical Flow to stabilize the "Region of Interest" during robot vibrations on rough terrain.
* **Multi-Obstacle Planner (VFH):** A Vector Field Histogram approach to find the safest path among multiple obstacles.
* **Watchdog Safety:** A dedicated thread that stops the robot instantly if the vision script freezes.

###  Body (Control System - Arduino/C++)
* **Zigzag Sweep Algorithm:** Automated coverage path planning to ensure 100% area scanning.
* **Sensor Fusion (IMU):** Combines MPU9250 Gyroscope and Magnetometer for drift-free heading.
* **Dual Feedback:** Real-time coordinates ($x, y, \theta$) calculated via high-resolution encoders.
* **Metal Detection Filter:** Intelligent signal processing to reduce false alarms from scrap metal.

---

## System Architecture



1.  **Perception Layer:** Camera feed processed by Jetson/Laptop running the Python V4 script.
2.  **Decision Layer:** VFH Planner determines the movement command (Forward, Left, Right, Stop).
3.  **Execution Layer:** Arduino Mega receives commands via Serial and manages motor PWM and PID.
4.  **Feedback Loop:** Encoders and IMU data sent back to the Brain for localization.

---

##  Getting Started

### Prerequisites
* **Hardware:** Arduino Mega, MPU9250 IMU, L298N/L293D Motor Drivers, Encoders, Metal Sensor.
* **Software:** Python 3.9+, OpenCV, Ultralytics (YOLO), PySerial.

### Installation
1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/yourusername/mine-clearing-robot.git](https://github.com/yourusername/mine-clearing-robot.git)
    ```
2.  **Setup Vision Environment:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Upload Firmware:**
    Open `Robot_Firmware.ino` in Arduino IDE and upload to your Mega.

4.  **Run the System:**
    ```bash
    python main_vision_v4.py --config config.json
    ```

---

##  Project Roadmap
- [x] Initial Zigzag navigation logic.
- [x] YOLOv8 integration for obstacle avoidance.
- [x] Adaptive Confidence & Dynamic ROI.
- [ ] **Next Step:** Development of a Web-based Dashboard for real-time monitoring.
- [ ] **Next Step:** Integration of GPS for outdoor large-scale mapping.

---

##  Contributing
We are building this to save lives. If you have experience in **Path Planning**, **Computer Vision**, or **Embedded Systems**, feel free to open an issue or a PR.

---

##  License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

