# Pedestrian-Detection-Tracking-and-Path-Prediction-Ultrasonic

Pedestrian Detection, Tracking & 1-Second Path Prediction (Ultrasonic Point Clouds)
This repo contains my MSc project exploring whether LiDAR-style perception methods can be adapted to ultrasonic time-of-flight data for pedestrian detection, tracking, and short-term path prediction.
Data are captured with the Calyo Pulse ultrasonic sensor and exported via SENSUS Viewer as grayscale PNG frames. Processing, visualization, and ML training are done in Google Colab (OpenCV, scikit-learn, PyTorch).

What this project does
Detects persistent structure in range–azimuth images using adaptive thresholding + temporal scoring.
Clusters with DBSCAN to extract objects and builds proportional occlusion masks.
Tracks pedestrians (multi-ID) using nearest-neighbour association with confirmation/miss logic and an ID-protection radius.
Predicts +1 s positions using an LSTM trained on (x, y, vx, vy) sequences.
Overlays predictions on the video: at 1 s anchors, draws a line from the current centroid to the model’s +1 s point; each line persists for 4 frames before a new one is allowed for that ID.

You’ll also find:
A dissertation PDF that explains the full pipeline and results.
A short MP4 presentation that shows the expected pipeline output.
A text file with the recordings folder link to the PNG frames used in the demo.

Repo layout
FinalPipeline.ipynb – Main Colab notebook: loads PNG frames, runs scoring → DBSCAN → occlusion → tracking, writes annotated frames, centroid CSVs, and the output video.
P.csv – Per-ID prediction table used to draw the “centroid → +1 s” lines in the video (ped_id,time_s,pred_x_1s,pred_y_1s,...).
DataAquisitionCSV(1S).ipynb – Builds the +1 s labeled dataset (TestSet1.csv) from tracked IDs.
DataAquisitionGraph.ipynb – Visualizes simulated trajectories to verify they look like typical walking behavior (straight segments, gentle turns, pauses).
LSTM(1S)Boxes.ipynb – Trains/evaluates the LSTM for +1 s prediction and makes tolerance-ring plots.
TrainingSet.csv, ValidationSet.csv – Datasets used to train/validate the LSTM.
TestSet1.csv – Test set built from real recordings (labels present when the same ID exists +1 s later).
lstm_fullseq_1s_4hz_model.pt – Trained PyTorch weights (1 s horizon, 4 Hz).
NOISE-IDS.png, PED-IDS.png – Figures summarizing excluded noise IDs and valid pedestrian IDs.
Pedestrian detection, tracking a… – Dissertation PDF.
Presentation.mp4 – Short video walkthrough of the final pipeline and results.
RECORDINGS FOLDER LINK.txt – Path/link to the raw PNG frames used by the pipeline.
README.md, LICENSE, .gitignore – Project docs & housekeeping.

Data & naming
Frames are exported from SENSUS Viewer as grayscale PNGs with this filename pattern:
[YYYY-MM-DD HH-MM-SS-ms].png
Frame rate is 4 Hz (Δt = 0.25 s). In code, the first PNG is treated as time 0.00 s, frame index 0.

Quick start (Colab)
Open FinalPipeline.ipynb in Google Colab.
Mount Drive and set the folder with your PNGs (see RECORDINGS FOLDER LINK.txt).
Run all cells. The notebook will:
Produce annotated frames in /content/OCCLUSION_FRAMES
Save centroids.csv, ped_tracks.csv
Build objects_occluded.mp4 (4 Hz)

(Optional) Place P.csv in /content/ to enable prediction lines on the video (see next section).
Prediction overlay (P.csv)
Columns (minimum): ped_id, time_s, pred_x_1s, pred_y_1s
The notebook aligns time_s so the first row of P.csv corresponds to frame 0 (t = 0.00 s), then maps frames by frame = round((time_s - t0)/0.25).
At anchor frames (every 1 s = every 4 frames), if a matching confirmed track for ped_id is present:
Draw a line from the current centroid → predicted (+1 s) point.
Keep the line on screen for 4 frames; do not spawn a new line for that ID until it expires.
This prevents clutter (you won’t get a new line every 0.25 s).

Outputs
objects_occluded.mp4 – Annotated video at 4 Hz.
/content/OCCLUSION_FRAMES/ – Per-frame PNGs with red objects, occlusion boxes, blue tracks, and (optionally) prediction lines.
centroids.csv – Red object clusters and centroids per frame.
ped_tracks.csv – Confirmed pedestrian tracks with per-frame positions and speeds.
TestSet1.csv – Final test dataset (+1 s labels when available) for evaluating the LSTM.
Training & evaluation (LSTM)
Use DataAquisitionCSV(1S).ipynb to build/update datasets from recordings.
Open LSTM(1S)Boxes.ipynb to train/evaluate the +1 s predictor on TrainingSet.csv / ValidationSet.csv.
The trained model (lstm_fullseq_1s_4hz_model.pt) can be used to generate P.csv for overlays.

Presentation & paper
Dissertation PDF: full technical description, background, methodology, and results.
Presentation.mp4: short walkthrough of the final pipeline and its expected appearance using the same recording link images.

Requirements
Google Colab (recommended)
Python libs: opencv-python, numpy, pandas, scikit-learn, torch, matplotlib (Colab notebooks install/use these)
