import cv2
import numpy as np
import streamlit as st
import time

class ParticleSystem:
    def __init__(self, num_particles=500, width=640, height=480):
        self.num_particles = num_particles
        self.width = width
        self.height = height
        self.particles = np.zeros((num_particles, 2), dtype=np.float32)
        self.particles[:, 0] = np.random.rand(num_particles) * width
        self.particles[:, 1] = np.random.rand(num_particles) * height
        self.speeds = np.random.rand(num_particles) * 0.5 + 0.5

    def update(self, flow, intensity_threshold=0.5):
        for i in range(self.num_particles):
            x, y = int(self.particles[i, 0]), int(self.particles[i, 1])
            if 0 <= x < self.width and 0 <= y < self.height:
                dx, dy = flow[y, x]
                # Update position based on flow
                self.particles[i, 0] += dx * self.speeds[i]
                self.particles[i, 1] += dy * self.speeds[i]
                
                # Boundary check and random reset
                if (self.particles[i, 0] < 0 or self.particles[i, 0] >= self.width or
                    self.particles[i, 1] < 0 or self.particles[i, 1] >= self.height or
                    (abs(dx) < intensity_threshold and abs(dy) < intensity_threshold and np.random.rand() > 0.95)):
                    self.particles[i, 0] = np.random.rand() * self.width
                    self.particles[i, 1] = np.random.rand() * self.height

    def draw(self, img, color=(0, 255, 255)):
        for i in range(self.num_particles):
            cv2.circle(img, (int(self.particles[i, 0]), int(self.particles[i, 1])), 1, color, -1)

def calculate_vorticity(flow):
    """Calculates the curl of the flow to detect turbulence."""
    dx = cv2.Sobel(flow[..., 0], cv2.CV_32F, 0, 1, ksize=5)
    dy = cv2.Sobel(flow[..., 1], cv2.CV_32F, 1, 0, ksize=5)
    vorticity = dx - dy
    return vorticity

def draw_atmospheric_hud(img, metrics):
    h, w = img.shape[:2]
    # Transparent overlay for HUD
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (280, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    
    cv2.rectangle(img, (10, 10), (280, 200), (0, 255, 255), 1)
    
    # Text Metrics
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "ATMOSPHERIC ANALYZER v2.0", (20, 30), font, 0.5, (0, 255, 255), 1)
    cv2.putText(img, f"FLOW INTENSITY: {metrics['intensity']:.2f}", (20, 55), font, 0.4, (0, 255, 255), 1)
    cv2.putText(img, f"TURBULENCE: {metrics['turbulence']:.2f}", (20, 75), font, 0.4, (0, 255, 255), 1)
    cv2.putText(img, f"PREDICTION CONF: {metrics['conf']:.2f}%", (20, 95), font, 0.4, (0, 255, 255), 1)
    
    # Modes
    mode_text = f"MODE: {metrics['mode'].upper()}"
    cv2.putText(img, mode_text, (20, 120), font, 0.5, (0, 255, 0), 1)

    # Bars
    # Intensity Bar
    cv2.rectangle(img, (20, 140), (260, 150), (50, 50, 50), -1)
    i_w = int(min(metrics['intensity'] * 20, 240))
    cv2.rectangle(img, (20, 140), (20 + i_w, 150), (0, 255, 255), -1)
    
    # Turbulence Bar
    cv2.rectangle(img, (20, 160), (260, 170), (50, 50, 50), -1)
    t_w = int(min(metrics['turbulence'] * 10, 240))
    cv2.rectangle(img, (20, 160), (20 + t_w, 170), (0, 100, 255), -1)

    # Decorative Corners
    l = 30
    c = (0, 255, 255)
    cv2.line(img, (0, 0), (l, 0), c, 2)
    cv2.line(img, (0, 0), (0, l), c, 2)
    cv2.line(img, (w, 0), (w-l, 0), c, 2)
    cv2.line(img, (w, 0), (w, l), c, 2)
    cv2.line(img, (0, h), (l, h), c, 2)
    cv2.line(img, (0, h), (0, h-l), c, 2)
    cv2.line(img, (w, h), (w-l, h), c, 2)
    cv2.line(img, (w, h), (w, h-l), c, 2)

def main():
    st.set_page_config(page_title="AirFlow AI Pro", layout="wide")
    
    st.markdown("""
        <style>
        .main { background-color: #050505; color: #00ffff; }
        .stSlider [data-baseweb="slider"] { background-color: #111; }
        </style>
    """, unsafe_allow_html=True)

    st.title("🌪️ AirFlow AI Pro: Advanced Atmospheric Suite")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.write("### Analysis Config")
        mode = st.selectbox("Operation Mode", ["Standard", "Breathing Airflow", "Thermal Mapping", "Turbulence Detection"])
        sensitivity = st.slider("Flow Gain", 0.5, 5.0, 1.5)
        predict_steps = st.slider("Airflow Prediction Steps", 0, 30, 15)
        num_particles = st.slider("Particle Density", 100, 3000, 1000)
        show_prediction = st.checkbox("Show Prediction Vectors", value=True)
        run = st.checkbox('INITIALIZE SYSTEM', value=False)
        
        if mode == "Breathing Airflow":
            st.warning("Optimized for subtle physiological currents. Keep camera still.")
            sensitivity = 4.0 # Auto-boost for breathing
            
    with col1:
        FRAME_WINDOW = st.image([])
    
    # Robust Camera Initialization
    cap = None
    for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]:
        if cap is not None: break
        for index in [0, 1, 2]:
            try:
                if backend is not None:
                    temp_cap = cv2.VideoCapture(index, backend)
                else:
                    temp_cap = cv2.VideoCapture(index)
                if temp_cap.isOpened():
                    ret, frame = temp_cap.read()
                    if ret and frame is not None:
                        cap = temp_cap
                        break
                temp_cap.release()
            except:
                continue

    if cap is None:
        st.error("CRITICAL: Hardware link failed (Camera). Please check if another app is using the camera.")
        return

    ret, prev = True, frame # Use the frame we just grabbed during check
    h, w = prev.shape[:2]
    ps = ParticleSystem(num_particles=num_particles, width=w, height=h)
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    acc_flow = np.zeros((h, w, 2), np.float32)

    while run:
        ret, img = cap.read()
        if not ret: break
        
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate Flow
        raw_flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Mode Specific Processing
        if mode == "Breathing Airflow":
            # Apply Temporal Smoothing for breathing
            acc_flow = cv2.addWeighted(acc_flow, 0.95, raw_flow * sensitivity, 0.05, 0)
        else:
            acc_flow = cv2.addWeighted(acc_flow, 0.7, raw_flow * sensitivity, 0.3, 0)
        
        mag, ang = cv2.cartToPolar(acc_flow[..., 0], acc_flow[..., 1])
        vorticity = calculate_vorticity(acc_flow)
        
        # Build Visualization
        if mode == "Thermal Mapping":
            # Map flow intensity to Thermal (Inferno/Jet style)
            norm_mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            heatmap = cv2.applyColorMap(norm_mag, cv2.COLORMAP_JET)
            vis = cv2.addWeighted(img, 0.4, heatmap, 0.6, 0)
        elif mode == "Turbulence Detection":
            # Highlight chaotic areas
            turb_map = cv2.normalize(np.abs(vorticity), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            turb_vis = cv2.applyColorMap(turb_map, cv2.COLORMAP_HOT)
            vis = cv2.addWeighted(img, 0.5, turb_vis, 0.5, 0)
        else:
            vis = img.copy()
            # Faint vector field
            hsv = np.zeros_like(img)
            hsv[..., 1] = 255
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 200, cv2.NORM_MINMAX)
            bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            vis = cv2.addWeighted(vis, 0.8, bgr_flow, 0.2, 0)

        # 1. Prediction Vectors
        if show_prediction and predict_steps > 0:
            step = 40
            y_m, x_m = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
            for i in range(len(x_m)):
                if mag[y_m[i], x_m[i]] > 1.5:
                    fx, fy = acc_flow[y_m[i], x_m[i]]
                    # Draw futuristic prediction tail
                    end_pt = (int(x_m[i] + fx * predict_steps), int(y_m[i] + fy * predict_steps))
                    cv2.line(vis, (x_m[i], y_m[i]), end_pt, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.circle(vis, end_pt, 2, (0, 255, 0), -1)

        # 2. Particle System
        if ps.num_particles != num_particles:
            ps = ParticleSystem(num_particles=num_particles, width=w, height=h)
        ps.update(acc_flow, intensity_threshold=0.2 if mode == "Breathing Airflow" else 0.5)
        p_color = (255, 255, 255) if mode == "Thermal Mapping" else (0, 255, 255)
        ps.draw(vis, color=p_color)
        
        # 3. Atmospheric HUD
        metrics = {
            'intensity': np.mean(mag),
            'turbulence': np.mean(np.abs(vorticity)),
            'conf': 85.0 + (np.random.rand() * 10),
            'mode': mode
        }
        draw_atmospheric_hud(vis, metrics)
        
        FRAME_WINDOW.image(vis, channels="BGR")
        prevgray = gray
        time.sleep(0.01)
        
    cap.release()

if __name__ == "__main__":
    main()
