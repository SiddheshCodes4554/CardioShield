CardioShield: Multimodal AI for Cardiovascular Risk Intelligence
Cardiovascular Disease (CVD) remains the leading cause of death globally. CardioShield is an AI-powered platform designed to enhance early detection by bridging the gap between clinical biomarkers and electrophysiological signals.
+1

🌟 The Innovation
Unlike standard diagnostic systems that rely on static data, CardioShield utilizes a Multimodal Fusion Engine. It combines:


Clinical Risk Modeling: Analyzing age, BP, and cholesterol using Gradient Boosting and LightGBM.
+1


ECG Deep Learning: An unsupervised Convolutional Autoencoder that detects anomalies in raw waveforms.

🛠️ Technical Implementation
1. Clinical Analytics

Algorithms: Evaluated Gradient Boosting, XGBoost, and LightGBM.


Performance: LightGBM delivered the best performance with an accuracy of approximately 74% and a ROC AUC near 0.80.


Key Predictors: Systolic blood pressure, cholesterol interaction, and age were identified as dominant predictors.

2. ECG Intelligence

Architecture: Convolutional Autoencoder designed for unsupervised anomaly detection.


Method: ECG signals were normalized, segmented, and reconstructed through the neural network.


Metric: Reconstruction Error (MSE) was used to quantify electrophysiological abnormalities.

3. Explainability & Visualization

Interpretability: Integrated SHAP-based feature attribution for clinical predictions.


Visualization: Heatmap visualizations and signal reconstruction overlays highlight abnormal waveform regions to improve clinical transparency.

🚀 Deployment
The system is deployed via a Streamlit dashboard, simulating a real-world clinical environment.


Functionality: Supports patient data input, ECG upload, risk scoring, and fusion intelligence reporting.

📈 Future Roadmap
Real-time ECG streaming and multi-lead signal modeling.

Integration with Electronic Health Records (EHR) and large-scale hospital deployment.

Developed for the Byte 2 Beat Hackathon.
