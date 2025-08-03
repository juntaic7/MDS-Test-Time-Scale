import numpy as np
import matplotlib.pyplot as plt

def compute_cc_score(w: any, c: any, k: int = 10):
    w = np.asarray(w, dtype=float)
    c = np.asarray(c, dtype=float)
    return w / (1 + np.exp(-k * (c - 0.5)))

def plot_cc_score(save_path='pas_score_visualization.pdf'):
    
    plt.rcParams.update({
        'font.size': 14,  # Base font size
        'axes.labelsize': 16,  # Size of axis labels
        'axes.titlesize': 18,  # Size of title
        'legend.fontsize': 14,  # Size of legend text
    })
    
    # Create a meshgrid for w and c values
    w = np.linspace(0, 1, 100)
    c = np.linspace(0, 1, 100)
    W, C = np.meshgrid(w, c)
    
    # Calculate cc_score for each point
    Z = compute_cc_score(W, C)
    
    # Convert to numpy arrays for plotting
    W_np = np.asarray(W)
    C_np = np.asarray(C)
    Z_np = np.asarray(Z)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    surface = ax.plot_surface(W_np, C_np, Z_np, cmap='viridis', alpha=0.8)
    
    w_ref = np.linspace(0, 1, 100)
    c_ref = np.full_like(w_ref, 0.5)
    z_ref = compute_cc_score(w_ref, c_ref)  
    ax.plot3D(w_ref, c_ref, z_ref, 'r-', linewidth=3, label='c = 0.5')
    
    for w_val in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        c_line = np.linspace(0, 1, 100)
        z_line = compute_cc_score(w_val, c_line)
        w_points = np.full_like(c_line, w_val)
        ax.plot3D(w_points, c_line, z_line, 'r--', alpha=0.5)
    
    ax.set_xlabel('w (win rate)')
    ax.set_ylabel('c (consistency)')
    ax.set_zlabel('PAS_score')
    ax.set_title('PAS Score Function Visualization')
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
    ax.view_init(elev=20, azim=160)
    
    # Add legend
    ax.legend()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_cc_score()