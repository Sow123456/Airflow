import cv2
import numpy as np

def calculate_vorticity(flow):
    """Calculates the curl of the flow to detect turbulence."""
    dx = cv2.Sobel(flow[..., 0], cv2.CV_32F, 0, 1, ksize=5)
    dy = cv2.Sobel(flow[..., 1], cv2.CV_32F, 1, 0, ksize=5)
    vorticity = dx - dy
    return vorticity

def run_simulation(scenario_name, prev_frame, current_frame, sensitivity=1.5):
    print(f"\n--- SIMULATION: {scenario_name} ---")
    
    # Calculate Dense Optical Flow (Farneback)
    flow = cv2.calcOpticalFlowFarneback(prev_frame, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow *= sensitivity
    
    # Calculate Metrics
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    vorticity = calculate_vorticity(flow)
    
    avg_intensity = np.mean(mag)
    avg_turbulence = np.mean(np.abs(vorticity))
    
    print(f"Metrics Calculated:")
    print(f"  -> Average Flow Intensity (Magnitude): {avg_intensity:.4f}")
    print(f"  -> Average Turbulence (Vorticity): {avg_turbulence:.4f}")
    
    # Simulated HUD Outcomes based on modes
    print("\nExpected HUD Outcomes per Mode:")
    
    # 1. Thermal Mapping (Based on Magnitude)
    if avg_intensity > 2.0:
        print("  [Thermal Mapping]: RED/HOT (High velocity flow detected)")
    elif avg_intensity > 0.5:
        print("  [Thermal Mapping]: GREEN/WARM (Moderate flow detected)")
    else:
        print("  [Thermal Mapping]: BLUE/COLD (Minimal flow detected)")
        
    # 2. Turbulence Detection (Based on Vorticity)
    if avg_turbulence > 5.0:
        print("  [Turbulence Detection]: HIGH ALERT (Chaotic/Swirling motion detected)")
    else:
        print("  [Turbulence Detection]: STABLE (Linear or smooth flow)")
        
    # 3. Breathing Airflow (Sensitivity Check)
    breathing_sensitivity = 4.0
    b_flow = cv2.calcOpticalFlowFarneback(prev_frame, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0) * breathing_sensitivity
    b_mag, _ = cv2.cartToPolar(b_flow[..., 0], b_flow[..., 1])
    if np.mean(b_mag) > 0.3: # Lower threshold due to high gain
        print("  [Breathing Airflow]: DETECTED (Subtle motion amplified successfully)")
    else:
        print("  [Breathing Airflow]: UNDETECTED (Motion too faint even with gain)")

def generate_synthetic_frames():
    h, w = 100, 100
    
    # Base Image (Textured to allow optical flow to track)
    base_img = np.random.randint(50, 200, (h, w), dtype=np.uint8)
    
    # SCENARIO 1: Linear Flow (e.g., strong steady fan)
    # Move the entire texture to the right
    linear_img = np.roll(base_img, shift=5, axis=1)
    run_simulation("Linear Airflow (Steady Fan)", base_img, linear_img)

    # SCENARIO 2: High Turbulence / Swirling (e.g., vortex or spinning blades)
    # Rotate the image slightly around the center
    M = cv2.getRotationMatrix2D((w/2, h/2), 5, 1)
    swirl_img = cv2.warpAffine(base_img, M, (w, h))
    run_simulation("Swirling / High Turbulence", base_img, swirl_img)

    # SCENARIO 3: Subtle Breathing (Micro-movements)
    # Move the texture by only 1 pixel
    breath_img = np.roll(base_img, shift=1, axis=0)
    run_simulation("Subtle Breathing (Micro-motion)", base_img, breath_img)
    
    # SCENARIO 4: No Movement (Control)
    run_simulation("Static Environment (No Wind)", base_img, base_img.copy())

if __name__ == "__main__":
    generate_synthetic_frames()
