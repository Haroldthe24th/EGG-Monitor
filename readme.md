Markdown
# Real-Time EEG Seizure Monitor

## 📖 Project Overview
This project is an end-to-end Machine Learning web application designed to detect epileptic seizures from raw brainwave data in real time. 

Instead of operating on static files, the system simulates a real-world medical device. A deep learning model analyzes sequential EEG data via a WebSocket stream, while a reactive frontend visualizes the brainwaves and triggers clinical alerts the moment a seizure signature is detected.

## 🛠️ Development Roadmap (To-Do)

### Phase 1: Machine Learning & Model Development
- [x] Perform Exploratory Data Analysis (EDA) and visualize 1-second EEG sequences.
- [x] Restructure and stratify the Kaggle dataset into Train, Validation, and Test sets.
- [ ] Design and build an LSTM model.
- [ ] Train the model on binary classification.
- [ ] Evaluate model precision/recall on the Test set.
- [ ] Export the trained model weights.

### Phase 2: Backend API (FastAPI)
- [ ] Set up a Python FastAPI server.
- [ ] Load the pre-trained LSTM model into the server memory.
- [ ] Create a WebSocket endpoint to receive live data (row by row).
- [ ] Write the inference logic to return a "Seizure Probability Score".

### Phase 3: Frontend Command Center (Next.js)
- [ ] Build the UI layout (Real-time graph, Upload component, Control Panel, Alert Modal).
- [ ] Implement the "Hardware Simulator" (parsing a CSV locally and trickling 1 row per second).
- [ ] Connect the frontend to the backend WebSocket.
- [ ] Integrate a live-updating chart to visualize the streaming EEG voltages.
- [ ] Program the global state to trigger a visual alarm when the model detects a seizure.