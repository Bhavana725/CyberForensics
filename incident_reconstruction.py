# incident_reconstruction.py
import streamlit as st
import pandas as pd
import json
from datetime import datetime
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt

def show_reconstruction_section():
    st.write("### 🕰️ Incident Reconstruction")
    
    try:
        with open("forensic_logs.json", "r") as f:
            logs = json.load(f)
        
        if not logs:
            st.warning("No forensic data available for reconstruction")
            return
        
        reconstruction_type = st.radio("Reconstruction Type", 
                                        ["Tabular Incident Summary", 
                                         "Event Sequence Diagram", 
                                         "Event Clustering and Grouping", 
                                         "Event Frequency Analysis"])
        
        if reconstruction_type == "Tabular Incident Summary":
            show_tabular_summary(logs)
        elif reconstruction_type == "Event Sequence Diagram":
            show_event_sequence_diagram(logs)
        elif reconstruction_type == "Event Clustering and Grouping":
            show_event_clustering(logs)
        elif reconstruction_type == "Event Frequency Analysis":
            show_event_frequency_analysis(logs)
            
    except FileNotFoundError:
        st.warning("No forensic data available")
    except json.JSONDecodeError:
        st.error("Error: Invalid JSON format in forensic_logs. Please check the file.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

def show_tabular_summary(logs):
    st.write("#### Tabular Incident Summary")
    incident_data = []
    for log in logs:
        try:
            timestamp = datetime.strptime(log['timestamp'], "%Y-%m-%dT%H:%M:%S.%f")
        except ValueError as e:
            st.error(f"Invalid timestamp: {log['timestamp']}. Error: {e}")
            continue
        incident_data.append({
            "Timestamp": timestamp,
            "Event Type": log['detection_type'],
            "Threat Level": log['threat_level'],
            "Results": str(log['results'])
        })
    df = pd.DataFrame(incident_data)
    st.dataframe(df)

def show_event_sequence_diagram(logs):
    st.write("#### Event Sequence Diagram")
    
    G = nx.DiGraph()
    events = []
    
    for log in logs:
        try:
            timestamp = datetime.strptime(log['timestamp'], "%Y-%m-%dT%H:%M:%S.%f")
        except ValueError as e:
            st.error(f"Invalid timestamp: {log['timestamp']}. Error: {e}")
            continue
        events.append((timestamp, log['detection_type']))
    
    events.sort()
    
    for i in range(len(events) - 1):
        G.add_edge(events[i][1], events[i+1][1])
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    st.pyplot(plt)
    plt.clf() #Clears the current figure.

def show_event_clustering(logs):
    st.write("#### Event Clustering and Grouping")
    
    incident_data = []
    for log in logs:
        try:
            timestamp = datetime.strptime(log['timestamp'], "%Y-%m-%dT%H:%M:%S.%f")
        except ValueError as e:
            st.error(f"Invalid timestamp: {log['timestamp']}. Error: {e}")
            continue
        incident_data.append({
            "Timestamp": timestamp,
            "Event Type": log['detection_type'],
            "Threat Level": log['threat_level']
        })
    df = pd.DataFrame(incident_data)
    
    grouped = df.groupby('Event Type').size().reset_index(name='Count')
    st.dataframe(grouped)

def show_event_frequency_analysis(logs):
    st.write("#### Event Frequency Analysis")
    
    incident_data = []
    for log in logs:
        try:
            timestamp = datetime.strptime(log['timestamp'], "%Y-%m-%dT%H:%M:%S.%f")
        except ValueError as e:
            st.error(f"Invalid timestamp: {log['timestamp']}. Error: {e}")
            continue
        incident_data.append({
            "Timestamp": timestamp,
            "Event Type": log['detection_type']
        })
    df = pd.DataFrame(incident_data)
    
    df['Hour'] = df['Timestamp'].dt.hour
    frequency = df.groupby(['Hour', 'Event Type']).size().reset_index(name='Count')
    
    fig = px.bar(frequency, x='Hour', y='Count', color='Event Type', barmode='group')
    st.plotly_chart(fig)
