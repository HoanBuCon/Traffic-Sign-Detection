<h1 align="center">Vietnam Traffic Sign Detection</h1>
<h3 align="center">Our first Machine Learning, Deep Learning, and Computer Vision Project</h3>

<hr>

<!-- LƯU Ý: Nếu có &#x26; thì phải sửa thành & -->
<h3>Contributors:</h3>
<ul>
  <li><a href="https://github.com/davidislearninghowtocode" target="_blank">🧠 <b>David Vu</b></a></li>
  <li><a href="https://github.com/AnDpTri" target="_blank">💻 <b>The Peak</b></a></li>
  <li><a href="https://github.com/HoanBuCon" target="_blank">💻 <b>Hoàn Bự Con</b></a></li>
</ul>

<hr>

<h3>About the Project:</h3>  

- **Dataset:** <a href="https://www.kaggle.com/datasets/maitam/vietnamese-traffic-signs" target="_blank">Vietnam Traffic Signs</a>

- The model is based on YOLOv8-medium architecture

- This project is designed with strong potential for future development and improvement.

- This shit is so peak

<hr>

<h3 align="left">Languages and Tools:</h3>
<p align="left">
  <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a>
  <a href="https://www.ultralytics.com/brand"> <img src="https://cdn.prod.website-files.com/680a070c3b99253410dd3dcf/680a070c3b99253410dd3e8d_UltralyticsYOLO_mark_blue.svg" alt="git" width="40" height="40"/> </a>
  <a href="https://pytorch.org/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/pytorch/pytorch-icon.svg" alt="pytorch" width="40" height="40"/> </a>
  <a href="https://git-scm.com/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/git-scm/git-scm-icon.svg" alt="git" width="40" height="40"/> </a>
</p>

<hr>

<h3>Installation guide:</h3>

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. (Optional) Create a virtual environment
```bash
python -m venv venv
# Activate virtual environment
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Create necessary folders
#### ❗Before running detection or training, make sure the following folders exist inside the `Traffic-Sign-Detection` directory:

- 📁 `input` — Place your input images here 
- 📁 `output` — Detected images will be saved here
- 📁 `real_time_output` — Detected videos will be saved here
- 📁 `all_weight` — Folder for storing trained model weights  
#### ❗The final path should look like:
- ```.\Traffic-Sign-Detection\input```
- ```.\Traffic-Sign-Detection\output```
- ```.\Traffic-Sign-Detection\real_time_output```
- ```.\Traffic-Sign-Detection\all_weight```

```text
└── Traffic-Sign-Detection\
    ├── input
    ├── output
    ├── real_time_output
    └── all_weight
```
#### ✅ Make sure these folders are created before running ```train.py``` or ```predict.py```.

### 5. Training
- Open ***Terminal*** and run this script:
```bash
python train.py
```

### 6. Detect traffic signs
#### Option 1: Detect with images
- Place the traffic signs images you want to detect into the 📁`input` folder: ```.\Traffic-Sign-Detection\input```
- Open ***Terminal*** and run the detection script:
```bash
python predict.py
```
#### Option 2: Detect with real time webcam/camera
- Open ***Terminal*** and run the script for basic real-time detection:
```bash
python real_time_predict.py
```

#### Option 3: Detect with real time webcam with advanced smooth
- Make sure you have placed `sort.py` in the `src` directory.
- Open ***Terminal*** and run the script for advanced real-time detection with SORT:
```bash
python real_time_predict_smooth_advanced.py
```

### 7. Data
- Detected images: ```.\Traffic-Sign-Detection\output```
- Detected videos: ```.\Traffic-Sign-Detection\real_time_output```
- Trained weights: ```.\Traffic-Sign-Detection\all_weight```

### 8. Enjoy🎉
- Thank you for checking out our project! Feel free to explore, improve, or contribute 🚀

<hr>

<h3 align="left">Connect with us:</h3>
<p align="left">
<a href="https://discord.gg/https://discord.gg/nckzdQE73u" target="_blank"><img align="center" src="https://upload.wikimedia.org/wikipedia/fr/4/4f/Discord_Logo_sans_texte.svg" alt="https://discord.gg/nckzdQE73u" height="30" width="40" /></a>
</p>