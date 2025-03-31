import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from scapy.all import sniff, IP, TCP, UDP
from sklearn.preprocessing import StandardScaler
import json
from datetime import datetime
import subprocess
import os
import requests

# Import forensic modules
from forensic_logging import show_logging_section
from threat_attribution import show_attribution_section
from incident_reconstruction import show_reconstruction_section

# ==========================
# 🚀 Load Trained Models & Preprocessing Setup
# ==========================

# DDoS Model Architecture
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# IDS Model Architecture
class DQN_IDS(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN_IDS, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load DDoS Model
dqn_ddos = DQN(input_dim=9, output_dim=2).to(device)
dqn_ddos.load_state_dict(torch.load("model_top9.pth", map_location=device))
dqn_ddos.eval()

# Load IDS Model
dqn_ids = DQN_IDS(input_dim=7, output_dim=2).to(device)
dqn_ids.load_state_dict(torch.load("dqn_threat_detection.pth", map_location=device))
dqn_ids.eval()

# Standard Scaler
scaler = StandardScaler()

# Feature Sets
ddos_features = ['Init_Win_bytes_forward', 'Fwd Packet Length Max', 'Subflow Fwd Bytes',
                'Bwd Packet Length Max', 'Fwd Packet Length Mean', 'Avg Fwd Segment Size',
                'Avg Bwd Segment Size', 'Total Length of Fwd Packets', 'Destination Port']

ids_features = ['dur', 'proto', 'service', 'state', 'sbytes', 'dbytes', 'rate']

captured_packets = []

# ==========================
# 🚀 Feature Extraction
# ==========================

def extract_ddos_features(packet):
    """Extract necessary features for DDoS detection"""
    features = {key: 0 for key in ddos_features}
    
    if packet.haslayer(IP):
        if packet.haslayer(TCP):
            features["Init_Win_bytes_forward"] = packet[TCP].window
            features["Destination Port"] = packet[TCP].dport
        elif packet.haslayer(UDP):
            features["Destination Port"] = packet[UDP].dport
        features["source_ip"] = packet[IP].src
        features["destination_ip"] = packet[IP].dst
    
    features["Total Length of Fwd Packets"] = len(packet)
    features["Fwd Packet Length Max"] = len(packet)
    features["Bwd Packet Length Max"] = len(packet)
    features["Fwd Packet Length Mean"] = len(packet)
    features["Avg Fwd Segment Size"] = len(packet)
    features["Avg Bwd Segment Size"] = len(packet)
    features["Subflow Fwd Bytes"] = len(packet)
    
    return features

def extract_ids_features(packet):
    """Extract necessary IDS features from network packet"""
    features = {key: 0 for key in ids_features}
    
    if packet.haslayer(IP):
        features["dur"] = 1
        features["proto"] = packet[IP].proto
        if packet.haslayer(TCP):
            features["service"] = 6
            features["state"] = 1
            features["sbytes"] = len(packet[TCP].payload)
            features["dbytes"] = len(packet) - len(packet[TCP].payload)
        elif packet.haslayer(UDP):
            features["service"] = 17
            features["state"] = 2
            features["sbytes"] = len(packet[UDP].payload)
            features["dbytes"] = len(packet) - len(packet[UDP].payload)
        features["source_ip"] = packet[IP].src
        features["destination_ip"] = packet[IP].dst
        
        features["rate"] = 1
    
    return features

def capture_traffic(extract_func):
    """Capture live network traffic and extract features"""
    global captured_packets
    captured_packets.clear()
    sniffed = sniff(count=10, timeout=5)
    captured_packets.extend([extract_func(pkt) for pkt in sniffed])

# ==========================
# 🚀 Dashboard Functions
# ==========================

def display_model_performance(title, metrics, confusion_matrix_name):
    """Display model performance metrics and confusion matrix image"""
    st.write(f"## 📊 {title}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 📈 Performance Metrics")
        df_report = pd.DataFrame(metrics, columns=["Metric", "Value"])
        st.table(df_report)
    
    with col2:
        st.write("### 🎯 Confusion Matrix")
        st.image(confusion_matrix_name, use_container_width=True)
    
    if "DDoS" in title:
        st.write("**Model Description**: This DDoS detection model uses deep Q-learning to identify distributed denial-of-service attacks with high accuracy.")
    else:
        st.write("**Model Description**: This intrusion detection system model employs deep learning to classify network traffic as allowed or blocked.")

def perform_detection(model, title, feature_set, extract_func):
    """Perform real-time detection with given model"""
    st.write(f"## ⚡ {title}")
    
    if st.button(f"Start {title.split()[0]} Analysis"):
        with st.spinner(f"Analyzing network traffic for {title.split()[0]}..."):
            capture_traffic(extract_func)
            
            if len(captured_packets) > 0:
                captured_df = pd.DataFrame(captured_packets)
                captured_df_for_model = captured_df[feature_set]
                
                scaled_features = scaler.fit_transform(captured_df_for_model)
                input_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(device)
                
                with torch.no_grad():
                    predictions = model(input_tensor)
                    predicted_labels = torch.argmax(predictions, axis=1).cpu().numpy()
                
                result_col = "Prediction" if "DDoS" in title else "Decision"
                result_values = ["Threat", "Normal"] if "DDoS" in title else ["Block", "Allow"]
                captured_df[result_col] = [result_values[x] for x in predicted_labels]
                
                # Update threat counts for DDoS detection
                if "DDoS" in title:
                    update_threat_counts(captured_df)
                    # Display threat counts right after detection
                    if hasattr(st.session_state, 'threat_ip_counts') and st.session_state.threat_ip_counts:
                        st.write("### 📊 Threat IP Counts")
                        counts_df = pd.DataFrame.from_dict(st.session_state.threat_ip_counts, 
                                                         orient='index', 
                                                         columns=['Threat Count'])
                        st.dataframe(counts_df.sort_values('Threat Count', ascending=False))
                
                # Create forensic logs
                log_data = []
                for index, row in captured_df.iterrows():
                    log_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "detection_type": title.split()[0],
                        "threat_level": "high" if row[result_col] == "Threat" or row[result_col] == "Block" else "low",
                        "results": row.to_dict(),
                        "model_type": "DDoS" if "DDoS" in title else "IDS"
                    }
                    log_data.append(log_entry)
                
                # Save logs to file
                logs_file_path = "forensic_logs.json"
                try:
                    if os.path.exists(logs_file_path):
                        with open(logs_file_path, "r") as f:
                            existing_logs = json.load(f)
                    else:
                        existing_logs = []
                except Exception as e:
                    st.error(f"Error reading log file: {e}")
                    existing_logs = []

                all_logs = existing_logs + log_data
                
                try:
                    with open(logs_file_path, "w") as f:
                        json.dump(all_logs, f, indent=4)
                    st.success("Forensic logs saved successfully!")
                except Exception as e:
                    st.error(f"Error saving logs: {e}")
                
                # Create download button
                try:
                    log_json = json.dumps(all_logs, indent=4)
                    st.download_button(
                        label="Download Forensic Logs",
                        data=log_json,
                        file_name="forensic_logs.json",
                        mime="application/json"
                    )
                except Exception as e:
                    st.error(f"Error creating download: {e}")
                
                # Display results
                st.write(f"### 🚨 {title.split()[0]} Results")
                st.dataframe(captured_df)
                
                # Visualization
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                sns.countplot(x=predicted_labels, ax=ax[0], palette="coolwarm")
                ax[0].set_xticklabels(result_values)
                ax[0].set_title("Distribution")
                
                positive_percentage = sum(predicted_labels)/len(predicted_labels)*100
                ax[1].pie([positive_percentage, 100-positive_percentage], 
                          labels=[result_values[1], result_values[0]], 
                          autopct='%1.1f%%',
                          colors=['#51cf66', '#ff6b6b'])
                ax[1].set_title(f"{result_values[1]} Percentage")
                st.pyplot(fig)

# ==========================
# 🚀 Main Dashboard
# ==========================

st.title("🚀 Real-Time Threat Detection Dashboard")

if 'threat_ip_counts' not in st.session_state:
    st.session_state.threat_ip_counts = {}
if 'blacklisted_ips' not in st.session_state:
    st.session_state.blacklisted_ips = set()

THREAT_THRESHOLD = 5

def update_threat_counts(detection_results):
    threats = detection_results[detection_results["Prediction"] == "Threat"]
    
    for _, threat in threats.iterrows():
        ip = threat.get('source_ip')
        if ip:
            if ip not in st.session_state.threat_ip_counts:
                st.session_state.threat_ip_counts[ip] = 0
            st.session_state.threat_ip_counts[ip] += 1
            
            if (st.session_state.threat_ip_counts[ip] >= THREAT_THRESHOLD and ip not in st.session_state.blacklisted_ips):
                st.session_state.blacklisted_ips.add(ip)
                st.error(f"🚨 ALERT: IP {ip} detected as threat {THREAT_THRESHOLD}+ times!")
                
                try:
                    subprocess.run(['sudo', 'iptables', '-A', 'INPUT', '-s', ip, '-j', 'DROP'], check=True)
                    st.success(f"IP {ip} has been automatically blocked!")
                except subprocess.CalledProcessError as e:
                    st.error(f"Failed to block IP {ip}: {str(e)}")

def get_threat_intel(ip_address):
    """Fetches threat intelligence for a given IP address."""
    api_key = "0c7d2b6ebafb911ee445a9bcd2012ea42e133e68955297f645f0f3016a909e08"
    api_url = f"https://www.virustotal.com/api/v3/ip_addresses/{ip_address}"
    headers = {"x-apikey": api_key}
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching threat intelligence: {e}")
        return None

def display_threat_intel_table(threat_data):
    """Displays threat intelligence results in a table."""
    if not threat_data or 'data' not in threat_data or 'attributes' not in threat_data['data'] or 'last_analysis_results' not in threat_data['data']['attributes']:
        st.warning("No detailed analysis results available.")
        return

    results = threat_data['data']['attributes']['last_analysis_results']
    table_data = []
    for engine, analysis in results.items():
        table_data.append({
            "Engine": engine,
            "Category": analysis.get("category", "N/A"),
            "Result": analysis.get("result", "N/A"),
            "Method": analysis.get("method", "N/A")
        })

    df = pd.DataFrame(table_data)
    st.table(df)

with st.sidebar:
    st.header("Navigation")
    main_selection = st.radio("Go to", ["Home", "Live Traffic Capture", "DDoS Model", "IDS Model", "Forensics", "Threat Intelligence"])
    
    if main_selection == "DDoS Model":
        ddos_selection = st.radio("DDoS Options", ["Real-time Detection", "Model Performance", "Threat Monitoring"])
    elif main_selection == "IDS Model":
        ids_selection = st.radio("IDS Options", ["Real-time Detection", "Model Performance"])
    elif main_selection == "Forensics":
        forensic_selection = st.radio("Forensics Options", ["Logging", "Threat Attribution", "Incident Reconstruction"])
    elif main_selection == "Threat Intelligence":
        threat_intel_section = st.radio("Threat Intel", ["IP Lookup"])  # Removed other options

if main_selection == "Home":
    st.write("### Welcome to the Real-Time Threat Detection Dashboard!")
    st.write("This dashboard captures live traffic, extracts features, and predicts threats.")
    st.write("Use the sidebar to navigate between different functionalities.")

elif main_selection == "Live Traffic Capture":
    st.write("## 🌐 Live Network Traffic Capture")
    col1, col2 = st.columns(2)
    with col1:
        feature_type = st.radio("Feature Type", ("DDoS Features", "IDS Features", "Both"))
    with col2:
        if st.button("Start Capturing Traffic"):
            with st.spinner("Capturing network traffic..."):
                captured_packets.clear()
                if feature_type == "DDoS Features":
                    capture_traffic(extract_ddos_features)
                elif feature_type == "IDS Features":
                    capture_traffic(extract_ids_features)
                else:
                    capture_traffic(lambda pkt: {**extract_ddos_features(pkt), **extract_ids_features(pkt)})
            st.success("✅ Traffic Captured!")
    if len(captured_packets) > 0:
        captured_df = pd.DataFrame(captured_packets)
        st.write("### 📊 Extracted Features from Live Traffic")
        st.dataframe(captured_df)

elif main_selection == "DDoS Model":
    if ddos_selection == "Real-time Detection":
        perform_detection(dqn_ddos, "DDoS Real-time Threat Detection", ddos_features, extract_ddos_features)
    elif ddos_selection == "Model Performance":
        display_model_performance("DDoS Model Performance", 
                                [["Accuracy", "99.80%"],
                                 ["Precision", "99.64%"],
                                 ["Recall", "100.00%"],
                                 ["F1-score", "99.82%"]], 
                                "confusion_matrix.png")
    elif ddos_selection == "Threat Monitoring":
        st.write("## 🛡️ Threat Monitoring Dashboard")
        st.write("### ⚠️ Current Threat Status")
        col1, col2 = st.columns(2)
        with col1:
            st.write("#### Threat Counts by IP")
            if st.session_state.threat_ip_counts:
                threat_df = pd.DataFrame.from_dict(st.session_state.threat_ip_counts, 
                                                 orient='index', 
                                                 columns=['Count'])
                st.dataframe(threat_df.style.highlight_max(axis=0))
            else:
                st.info("No threats detected yet")
        with col2:
            st.write("#### Blacklisted IPs")
            if st.session_state.blacklisted_ips:
                st.write(list(st.session_state.blacklisted_ips))
            else:
                st.info("No IPs blacklisted yet")
        if st.button("Clear All Threat Counts"):
            st.session_state.threat_ip_counts = {}
            st.session_state.blacklisted_ips = set()
            st.success("Threat counts and blacklist cleared!")

elif main_selection == "IDS Model":
    if ids_selection == "Real-time Detection":
        perform_detection(dqn_ids, "IDS Real-time Threat Detection", ids_features, extract_ids_features)
    else:
        display_model_performance("IDS Model Performance", 
                                [["Accuracy", "94.93%"],
                                 ["Precision", "97.63%"],
                                 ["Recall", "96.57%"],
                                 ["F1 Score", "97.10%"],
                                 ["Allow Rate", "47.49%"],
                                 ["Block Rate", "52.51%"]], 
                                "cf_ids.jpeg")

elif main_selection == "Forensics":
    st.write("## 🔍 Digital Forensics Center")
    if forensic_selection == "Logging":
        show_logging_section()
    elif forensic_selection == "Threat Attribution":
        show_attribution_section()
    elif forensic_selection == "Incident Reconstruction":
        show_reconstruction_section()

elif main_selection == "Threat Intelligence":
    if threat_intel_section == "IP Lookup":
        st.write("## 🔍 IP Threat Lookup")
        ip_to_lookup = st.text_input("Enter IP Address to Lookup:")
        if st.button("Lookup"):
            if ip_to_lookup:
                threat_data = get_threat_intel(ip_to_lookup)
                if threat_data:
                    st.write("### Threat Intelligence Results:")
                    st.write(f"Country: {threat_data['data']['attributes'].get('country', 'N/A')}")
                    st.write(f"Last Analysis Date: {datetime.fromtimestamp(threat_data['data']['attributes'].get('last_analysis_date', 0))}")
                    st.write("### Analysis Breakdown:")
                    display_threat_intel_table(threat_data)
                    st.write("### Raw JSON Data:")
                    st.json(threat_data)
                else:
                    st.warning("No threat intelligence found for this IP.")
