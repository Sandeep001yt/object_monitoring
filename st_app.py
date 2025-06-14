import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import time
import json
from pathlib import Path
import base64
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
# Page configuration
st.set_page_config(
    page_title="Object Detection Monitoring Tool",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
   
    .section-header {
        color: #2e86ab;
        font-size: 1.5rem;
        margin-bottom: 1rem;
        padding: 0.5rem;
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        border-left: 4px solid #1f77b4;
        border-radius: 5px;
    }
   
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
   
    .stButton > button {
        width: 100%;
        border-radius: 20px;
        border: none;
        padding: 0.5rem;
        transition: all 0.3s;
    }
   
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
   
    .video-controls {
        background: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
   
    .class-button-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 1rem 0;
    }
    
    /* Improved image and column alignment */
    .stImage {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
    }
    
    .stImage > img {
        max-width: 100% !important;
        max-height: 100% !important;
        object-fit: contain !important;
    }
    
    /* Fix column alignment */
    div[data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
        align-items: stretch !important;
    }
    
    /* Make metric containers uniform height */
    [data-testid="metric-container"] {
        height: 100px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        margin-top: 1rem;
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
    }
    
    /* Fix caption alignment */
    .stImage figcaption {
        text-align: center !important;
        width: 100% !important;
        margin-top: 0.5rem !important;
    }
    
    /* Ensure uniform column widths */
    div.row-widget.stHorizontal > div {
        flex: 1;
        width: 33.33% !important;
    }
    
    /* Fix image container heights */
    div[data-testid="column"] div[data-testid="stImage"] {
        height: 400px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    /* Fix warning message alignment */
    div.stAlert {
        width: 100% !important;
        text-align: center !important;
    }
</style>
""", unsafe_allow_html=True)

# Color mapping for different classes
CLASS_COLORS = {
    'car': '#4ECDC4',
    'bus': '#96CEB4',
    'truck': '#45B7D1',
    'motorcycle': '#FF6B6B'
}

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def draw_bounding_boxes(image, detection_data, selected_class='all_classes'):
    """Draw bounding boxes on image based on detection data"""
    if not detection_data:
        return image
   
    # Create a copy of the image
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
   
    # Try to use a better font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
   
    # Draw bounding boxes for each class
    for class_name, boxes in detection_data.items():
        if selected_class != 'all_classes' and selected_class != class_name:
            continue
           
        color = CLASS_COLORS.get(class_name, '#FFFFFF')
        rgb_color = hex_to_rgb(color)
       
        for box in boxes:
            # Box format is [xmin, xmax, ymin, ymax]
            xmin, xmax, ymin, ymax = box
           
            # Draw rectangle
            draw.rectangle([xmin, ymin, xmax, ymax], outline=rgb_color, width=3)
           
            # Draw label background
            label = class_name
            bbox = draw.textbbox((0, 0), label, font=font)
            label_width = bbox[2] - bbox[0]
            label_height = bbox[3] - bbox[1]
           
            draw.rectangle([xmin, ymin-label_height-4, xmin+label_width+8, ymin], fill=rgb_color)
            draw.text((xmin+4, ymin-label_height-2), label, fill='black', font=font)
   
    return img_with_boxes

def create_demo_dataframe():
    """Create a demo dataframe for testing"""
    demo_data = {
        'frame': ['frame_1', 'frame_2', 'frame_3'],
        'image_path': [
            'https://via.placeholder.com/640x480/FF6B6B/FFFFFF?text=Frame+1',
            'https://via.placeholder.com/640x480/4ECDC4/FFFFFF?text=Frame+2',
            'https://via.placeholder.com/640x480/45B7D1/FFFFFF?text=Frame+3'
        ],
        'detection_data': [
            {'motorcycle': [[100, 180, 120, 220], [200, 290, 250, 360]], 'car': [[50, 110, 80, 155]]},
            {'car': [[300, 370, 320, 405]], 'truck': [[150, 250, 180, 300]]},
            {'bus': [[250, 370, 280, 420]], 'motorcycle': [[350, 435, 370, 475]]}
        ],
        'missed_data': [
            {'car': [[80, 120, 100, 180]], 'motorcycle': [[120, 170, 140, 210]]},
            {'bus': [[180, 215, 200, 275]], 'truck': [[300, 320, 320, 380]]},
            {'car': [[350, 395, 370, 400]], 'motorcycle': [[400, 445, 420, 505]]}
        ]
    }
    return pd.DataFrame(demo_data)

def load_image_from_path(image_path):
    """Load image from path (supports both local paths and URLs)"""
    try:
        if image_path.startswith('http'):
            # Handle URL
            import requests
            response = requests.get(image_path)
            return Image.open(BytesIO(response.content))
        else:
            # Handle local path
            return Image.open(image_path)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        # Return a placeholder image
        return Image.new('RGB', (640, 480), color='gray')

def create_missed_detection_graphs(df):
    """
    Create graphs showing the number of missed detections per frame
    """
    # Initialize data storage
    frames = list(range(1, len(df) + 1))
    all_classes_missed = []
    class_missed = {
        'car': [],
        'bus': [],
        'truck': [],
        'motorcycle': []
    }
    
    # Extract data
    for _, row in df.iterrows():
        missed_data = row.get('missed_data', {})
        
        # Count all missed detections
        total_missed = sum(len(boxes) for boxes in missed_data.values())
        all_classes_missed.append(total_missed)
        
        # Count missed detections per class
        for class_name in class_missed.keys():
            class_missed[class_name].append(len(missed_data.get(class_name, [])))
    
    # Calculate averages
    avg_all_missed = sum(all_classes_missed) / len(all_classes_missed) if all_classes_missed else 0
    avg_class_missed = {
        class_name: sum(counts) / len(counts) if counts else 0
        for class_name, counts in class_missed.items()
    }
    
    # Create different graph types
    graphs = {}
    
    # 1. Line chart for all classes
    fig_line = px.line(
        x=frames,
        y=all_classes_missed,
        title="Missed Detections Over Time (All Classes)",
        labels={"x": "Frame Number", "y": "Number of Missed Detections"},
        template="plotly_white"
    )
    fig_line.add_hline(y=avg_all_missed, line_dash="dash", line_color="red",
                      annotation_text=f"Avg: {avg_all_missed:.2f}")
    graphs['line_all'] = fig_line
    
    # 2. Bar chart for all classes
    fig_bar = px.bar(
        x=frames,
        y=all_classes_missed,
        title="Missed Detections Per Frame (All Classes)",
        labels={"x": "Frame Number", "y": "Number of Missed Detections"},
        template="plotly_white"
    )
    fig_bar.add_hline(y=avg_all_missed, line_dash="dash", line_color="red",
                     annotation_text=f"Avg: {avg_all_missed:.2f}")
    graphs['bar_all'] = fig_bar
    
    # 3. Stacked area chart for class breakdown
    fig_area = go.Figure()
    for class_name, counts in class_missed.items():
        fig_area.add_trace(go.Scatter(
            x=frames,
            y=counts,
            mode='lines',
            stackgroup='one',
            name=class_name,
            line=dict(width=0.5),
            hoverinfo='name+y'
        ))
    fig_area.update_layout(
        title="Stacked Missed Detections by Class",
        xaxis_title="Frame Number",
        yaxis_title="Number of Missed Detections",
        template="plotly_white",
        hovermode="x unified"
    )
    graphs['stacked_area'] = fig_area
    
    # 4. Individual class line charts
    for class_name, counts in class_missed.items():
        fig = px.line(
            x=frames,
            y=counts,
            title=f"Missed Detections for {class_name.title()}",
            labels={"x": "Frame Number", "y": "Number of Missed Detections"},
            template="plotly_white"
        )
        fig.add_hline(y=avg_class_missed[class_name], line_dash="dash", line_color="red",
                     annotation_text=f"Avg: {avg_class_missed[class_name]:.2f}")
        graphs[f'line_{class_name}'] = fig
    
    # 5. Subplot with all individual classes
    fig_subplots = make_subplots(rows=2, cols=2, 
                                subplot_titles=("Car", "Bus", "Truck", "Motorcycle"))
    
    classes = list(class_missed.keys())
    for i, class_name in enumerate(classes):
        row = i // 2 + 1
        col = i % 2 + 1
        fig_subplots.add_trace(
            go.Scatter(x=frames, y=class_missed[class_name], name=class_name),
            row=row, col=col
        )
        fig_subplots.add_hline(y=avg_class_missed[class_name], line_dash="dash", line_color="red",
                              row=row, col=col)
    
    fig_subplots.update_layout(
        title_text="Missed Detections by Class",
        height=800,
        showlegend=False
    )
    graphs['subplots'] = fig_subplots
    
    return graphs, avg_all_missed, avg_class_missed

def main():
    # Title
    st.markdown('<h1 class="main-header">🎯 Object Detection Monitoring Tool</h1>', unsafe_allow_html=True)
   
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'current_frame_idx' not in st.session_state:
        st.session_state.current_frame_idx = 0
    if 'video_playing' not in st.session_state:
        st.session_state.video_playing = False
    if 'detection_class' not in st.session_state:
        st.session_state.detection_class = 'all_classes'
    if 'missed_class' not in st.session_state:
        st.session_state.missed_class = 'all_classes'
   
    # Sidebar for data input and controls
    with st.sidebar:
        st.markdown('<h2 style="color: #1f77b4;">📊 Data Input</h2>', unsafe_allow_html=True)
       
        # File upload
        uploaded_file = st.file_uploader(
            "Upload DataFrame (CSV/Pickle)",
            type=['csv', 'pkl', 'pickle'],
            help="Upload a dataframe with columns: frame, image_path, detection_data, missed_data"
        )
       
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    # Convert string representations of dictionaries back to dictionaries
                    if 'detection_data' in df.columns:
                        df['detection_data'] = df['detection_data'].apply(lambda x: eval(x) if isinstance(x, str) else x)
                    if 'missed_data' in df.columns:
                        df['missed_data'] = df['missed_data'].apply(lambda x: eval(x) if isinstance(x, str) else x)
                else:
                    df = pd.read_pickle(uploaded_file)
               
                st.session_state.df = df
                st.success("✅ DataFrame loaded successfully!")
                st.write(f"📝 Loaded {len(df)} frames")
               
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
       
        # Demo data button
        if st.button("🎮 Load Demo Data", type="secondary"):
            st.session_state.df = create_demo_dataframe()
            st.session_state.current_frame_idx = 0
            st.success("✅ Demo data loaded!")
       
        # Show dataframe info if loaded
        if st.session_state.df is not None:
            st.markdown("---")
            st.markdown("**📋 DataFrame Info:**")
            df = st.session_state.df
            st.write(f"• Frames: {len(df)}")
            st.write(f"• Columns: {list(df.columns)}")
           
            with st.expander("🔍 View DataFrame"):
                st.dataframe(df)
   
    # Main content area
    if st.session_state.df is not None:
        df = st.session_state.df
       
        # Create three columns for the main layout with equal widths
        col1, col2, col3 = st.columns([1, 1, 1])
       
        with col1:
            st.markdown('<div class="section-header">📷 Original Frame</div>', unsafe_allow_html=True)
            # Display current frame
            current_row = df.iloc[st.session_state.current_frame_idx]
            original_image = load_image_from_path(current_row['image_path'])
            st.image(original_image, caption=f"Frame: {current_row['frame']}", use_container_width=True)
            # Frame navigation controls
            frame_col1, frame_col2, frame_col3 = st.columns([1, 2, 1])
           
            with frame_col1:
                if st.button("⬅️ Previous", key="prev_frame"):
                    if st.session_state.current_frame_idx > 0:
                        st.session_state.current_frame_idx -= 1
                    else:
                        st.session_state.current_frame_idx = len(df) - 1
           
            with frame_col2:
                st.markdown(f'<div style="text-align: center; padding: 8px; background: #f0f8ff; border-radius: 5px;">Frame {st.session_state.current_frame_idx + 1} / {len(df)}</div>', unsafe_allow_html=True)
           
            with frame_col3:
                if st.button("Next ➡️", key="next_frame"):
                    if st.session_state.current_frame_idx < len(df) - 1:
                        st.session_state.current_frame_idx += 1
                    else:
                        st.session_state.current_frame_idx = 0
           
            # Video controls
            st.markdown('<div class="video-controls">', unsafe_allow_html=True)
           
            video_col1, video_col2 = st.columns([1, 1])
            with video_col1:
                if st.button("▶️ Play Video" if not st.session_state.video_playing else "⏸️ Stop Video", key="video_toggle"):
                    st.session_state.video_playing = not st.session_state.video_playing
           
            with video_col2:
                frame_rate = st.slider("Frame Rate (FPS)", 0.5, 10.0, 2.0, 0.5, key="frame_rate")
           
            st.markdown('</div>', unsafe_allow_html=True)
           

       
        with col2:
            st.markdown('<div class="section-header">🎯 Object Detection Output</div>', unsafe_allow_html=True)
            # Display image with detection boxes
            detection_data = current_row.get('detection_data', {})
            if detection_data:
                detection_image = draw_bounding_boxes(
                    original_image,
                    detection_data,
                    st.session_state.detection_class
                )
                st.image(detection_image, caption="Detection Results", use_container_width=True)
               
                # Show detection statistics
                if st.session_state.detection_class == 'all_classes':
                    total_detections = sum(len(boxes) for boxes in detection_data.values())
                    st.metric("Total Detections", total_detections)
                else:
                    class_detections = len(detection_data.get(st.session_state.detection_class, []))
                    st.metric(f"{st.session_state.detection_class.title()} Detections", class_detections)
            else:
                st.image(original_image, caption="No Detection Data", use_container_width=True)
                st.warning("⚠️ No detection data available for this frame")
 
            # Class selection buttons for detection
            st.markdown("**Select Class to Display:**")
            detection_classes = ['all_classes', 'car', 'bus', 'truck', 'motorcycle']
           
            # Create buttons in rows
            det_cols = st.columns(len(detection_classes))
            for i, class_name in enumerate(detection_classes):
                with det_cols[i]:
                    if st.button(class_name.replace('_', ' ').title(), key=f"det_{class_name}"):
                        st.session_state.detection_class = class_name
           
            # Show current selection
            st.info(f"🎯 Current Selection: {st.session_state.detection_class.replace('_', ' ').title()}")
           
       
        with col3:
            st.markdown('<div class="section-header">🔍 Missed Classes</div>', unsafe_allow_html=True)
            # Display image with missed detection boxes
            missed_data = current_row.get('missed_data', {})
            if missed_data:
                missed_image = draw_bounding_boxes(
                    original_image,
                    missed_data,
                    st.session_state.missed_class
                )
                st.image(missed_image, caption="Missed Detections", use_container_width=True)
               
                # Show missed detection statistics
                if st.session_state.missed_class == 'all_classes':
                    total_missed = sum(len(boxes) for boxes in missed_data.values())
                    st.metric("Total Missed", total_missed)
                else:
                    class_missed = len(missed_data.get(st.session_state.missed_class, []))
                    st.metric(f"{st.session_state.missed_class.title()} Missed", class_missed)
            else:
                st.image(original_image, caption="No Missed Detection Data", use_container_width=True)
                st.warning("⚠️ No missed detection data available for this frame")
            # Class selection buttons for missed classes
            st.markdown("**Select Class to Display:**")
            missed_classes = ['all_classes', 'car', 'bus', 'truck', 'motorcycle']
           
            # Create buttons in rows  
            missed_cols = st.columns(len(missed_classes))
            for i, class_name in enumerate(missed_classes):
                with missed_cols[i]:
                    if st.button(class_name.replace('_', ' ').title(), key=f"missed_{class_name}"):
                        st.session_state.missed_class = class_name
           
            # Show current selection
            st.info(f"🔍 Current Selection: {st.session_state.missed_class.replace('_', ' ').title()}")
           

       
        # Video playback logic
        if st.session_state.video_playing:
            time.sleep(1.0 / frame_rate)
            if st.session_state.current_frame_idx < len(df) - 1:
                st.session_state.current_frame_idx += 1
            else:
                st.session_state.current_frame_idx = 0
            st.rerun()
       
        st.markdown("---")

    # Analytics section
    if st.session_state.df is not None:
        st.markdown('<h2 class="section-header" style="background: #f5f7ff; background-image: linear-gradient(120deg, #e0f2fe, #f1f5ff); border-left: 4px solid #3b82f6; color: black;">📈 Analytics Dashboard</h2>', unsafe_allow_html=True)



        # Create tabs for different analytics views
        analytics_tabs = st.tabs([
            "Overall Missed Detections", 
            "Class-specific Graphs",
            "Individual Classes", 
            "Advanced Visualization"
        ])
        
        # Generate the graphs
        graphs, avg_all_missed, avg_class_missed = create_missed_detection_graphs(st.session_state.df)
        
        with analytics_tabs[0]:
            # Overall Missed Detections
            graph_type = st.radio(
                "Select Graph Type",
                options=["Line Chart", "Bar Chart", "Stacked Area Chart"],
                horizontal=True,
                key="overall_graph_type"
            )
            
            if graph_type == "Line Chart":
                st.plotly_chart(graphs['line_all'], use_container_width=True)
            elif graph_type == "Bar Chart":
                st.plotly_chart(graphs['bar_all'], use_container_width=True)
            else:  # Stacked Area Chart
                st.plotly_chart(graphs['stacked_area'], use_container_width=True)
                
            # Display overall average
            st.metric(
                "Average Missed Detections per Frame (All Classes)",
                f"{avg_all_missed:.2f}"
            )
        
        with analytics_tabs[1]:
            # Class-specific Graphs
            st.plotly_chart(graphs['subplots'], use_container_width=True)
            
            # Display averages for each class
            cols = st.columns(4)
            for i, (class_name, avg) in enumerate(avg_class_missed.items()):
                with cols[i]:
                    st.metric(
                        f"Average {class_name.title()} Missed",
                        f"{avg:.2f}"
                    )
        
        with analytics_tabs[2]:
            # Individual class graphs
            selected_class = st.selectbox(
                "Select Class",
                options=["car", "bus", "truck", "motorcycle"],
                key="individual_class_graph"
            )
            
            st.plotly_chart(graphs[f'line_{selected_class}'], use_container_width=True)
            
            # Show statistics for selected class
            st.metric(
                f"Average {selected_class.title()} Missed per Frame",
                f"{avg_class_missed[selected_class]:.2f}"
            )
            
        with analytics_tabs[3]:
            # Advanced visualization
            st.markdown("### Advanced Visualization Options")
            
            # Options for advanced visualizations
            chart_type = st.selectbox(
                "Chart Type",
                ["Line", "Bar", "Area", "Scatter"],
                key="adv_chart_type"
            )
            
            rolling_window = st.slider(
                "Rolling Average Window",
                min_value=1,
                max_value=10,
                value=1,
                key="rolling_window"
            )
            
            cumulative = st.checkbox("Show Cumulative Values", key="show_cumulative")
            
            # Extract data again for custom processing
            frames = list(range(1, len(st.session_state.df) + 1))
            data = {}
            
            selected_classes = st.multiselect(
                "Select Classes to Display",
                options=["car", "bus", "truck", "motorcycle", "all_classes"],
                default=["all_classes"],
                key="adv_selected_classes"
            )
            
            for _, row in st.session_state.df.iterrows():
                missed_data = row.get('missed_data', {})
                
                # Process all classes if selected
                if "all_classes" in selected_classes:
                    if "all_classes" not in data:
                        data["all_classes"] = []
                    total_missed = sum(len(boxes) for boxes in missed_data.values())
                    data["all_classes"].append(total_missed)
                
                # Process individual selected classes
                for class_name in selected_classes:
                    if class_name == "all_classes":
                        continue
                        
                    if class_name not in data:
                        data[class_name] = []
                    
                    class_missed = len(missed_data.get(class_name, []))
                    data[class_name].append(class_missed)
            
            # Process data with rolling window if needed
            if rolling_window > 1:
                for class_name in data:
                    # Calculate rolling average
                    values = data[class_name]
                    rolling_values = []
                    for i in range(len(values)):
                        start = max(0, i - rolling_window + 1)
                        window = values[start:i+1]
                        rolling_values.append(sum(window) / len(window))
                    data[class_name] = rolling_values
            
            # Calculate cumulative values if needed
            if cumulative:
                for class_name in data:
                    cumulative_values = []
                    current_sum = 0
                    for value in data[class_name]:
                        current_sum += value
                        cumulative_values.append(current_sum)
                    data[class_name] = cumulative_values
            
            # Create custom chart based on selections
            fig = go.Figure()
            
            for class_name, values in data.items():
                if chart_type == "Line":
                    fig.add_trace(go.Scatter(
                        x=frames, y=values, mode='lines', name=class_name
                    ))
                elif chart_type == "Bar":
                    fig.add_trace(go.Bar(
                        x=frames, y=values, name=class_name
                    ))
                elif chart_type == "Area":
                    fig.add_trace(go.Scatter(
                        x=frames, y=values, mode='lines', name=class_name,
                        fill='tozeroy'
                    ))
                elif chart_type == "Scatter":
                    fig.add_trace(go.Scatter(
                        x=frames, y=values, mode='markers', name=class_name
                    ))
            
            # Update layout
            title_suffix = ""
            if rolling_window > 1:
                title_suffix += f" (Rolling Avg: {rolling_window})"
            if cumulative:
                title_suffix += " (Cumulative)"
                
            fig.update_layout(
                title=f"Missed Detections Analysis{title_suffix}",
                xaxis_title="Frame Number",
                yaxis_title="Number of Missed Detections",
                template="plotly_white",
                legend_title="Classes"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add download button for the chart
            st.download_button(
                label="Download Chart as HTML",
                data=fig.to_html(),
                file_name="missed_detections_chart.html",
                mime="text/html"
            )
        st.markdown("---")
        # Additional information section
        info_col1, info_col2, info_col3 = st.columns(3)
       
        with info_col1:
            st.markdown("### 📊 Current Frame Stats")
            st.json({
                "frame_name": current_row['frame'],
                "detection_classes": list(detection_data.keys()) if detection_data else [],
                "missed_classes": list(missed_data.keys()) if missed_data else []
            })
       
        with info_col2:
            st.markdown("### 🎨 Class Colors")
            for class_name, color in CLASS_COLORS.items():
                st.markdown(f'<div style="display: flex; align-items: center;"><div style="width: 20px; height: 20px; background-color: {color}; margin-right: 10px; border-radius: 3px;"></div>{class_name}</div>', unsafe_allow_html=True)
       
        with info_col3:
            st.markdown("### ⚙️ Controls")
            st.markdown("""
            - **Navigation**: Use Previous/Next buttons or video mode
            - **Video Mode**: Toggle play/pause with customizable frame rate
            - **Class Filters**: Click class buttons to show specific detections
            - **All Classes**: Default view showing all detections
            """)
   
    else:
        # Welcome screen
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin: 2rem 0;">
            <h2>🚀 Welcome to Object Detection Visualizer</h2>
            <p style="font-size: 1.1rem; margin: 1rem 0;">Upload your dataframe or load demo data to get started!</p>
            <p>Your dataframe should contain columns: <code>frame</code>, <code>image_path</code>, <code>detection_data</code>, <code>missed_data</code></p>
        </div>
        """, unsafe_allow_html=True)
       
        # Expected dataframe format
        st.markdown("### 📋 Expected DataFrame Format")
       
        sample_df = pd.DataFrame({
            'frame': ['frame_1', 'frame_2'],
            'image_path': ['/path/to/image1.jpg', '/path/to/image2.jpg'],
            'detection_data': [
                "{'motorcycle': [[100, 180, 120, 220]], 'car': [[50, 110, 80, 155]]}",
                "{'car': [[300, 370, 320, 405]], 'truck': [[150, 250, 180, 300]]}"
            ],
            'missed_data': [
                "{'car': [[80, 120, 100, 180]]}",
                "{'motorcycle': [[120, 170, 140, 210]]}"
            ]
        })
       
        st.dataframe(sample_df)
       
        st.markdown("""
        **Column Descriptions:**
        - `frame`: Frame identifier (e.g., frame_1, frame_2, ...)
        - `image_path`: Path to the image file (local path or URL)
        - `detection_data`: Dictionary with class names as keys and list of bounding boxes as values
        - `missed_data`: Dictionary with missed class names and their bounding boxes
       
        **Bounding Box Format:** `[xmin, xmax, ymin, ymax]`
        """)

if __name__ == "__main__":
    main()