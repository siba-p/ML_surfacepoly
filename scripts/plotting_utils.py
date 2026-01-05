# plotting_utils.py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker

# ------------------------ Plotting Configuration ------------------------ #
LARGE_SIZE = 14
MEDIUM_SIZE = 12
SMALL_SIZE = 6

params = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],# Use serif fonts like Times New Roman
    'font.size': 6,
    'font.weight': 'medium',
    'figure.figsize': (3.5, 2.5),
    'axes.titlesize': 14,
    'axes.labelsize': SMALL_SIZE,
    'xtick.labelsize': SMALL_SIZE,
    'ytick.labelsize': SMALL_SIZE,
    'axes.labelweight': 'medium',
    'legend.fontsize': SMALL_SIZE,
    'legend.frameon': True,
    'legend.loc': 'best',
    'lines.linewidth': 1,
    'lines.markersize': 6,
    'grid.alpha': 0.5,
    'savefig.dpi': 400,
    'text.usetex': False,
    'axes.grid': False,
    'axes.grid.axis': 'both',
    'axes.grid.which': 'both',
    'figure.autolayout': True,
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 1,
    'xtick.minor.size': 1,
    'ytick.major.size': 1,
    'ytick.minor.size': 1,
}
plt.rcParams['pdf.fonttype'] = 42
def update_params():
    plt.rcParams.update(params)
def visualize_hex(hex_code):
    rgb = mcolors.to_rgb(hex_code)
    r, g, b = [round(x * 255) for x in rgb]

    print(f"\n Hex: {hex_code}")
    print(f" RGB: {r}, {g}, {b}")
    print(f" Red = {r}, Green = {g}, Blue = {b}")

    fig, ax = plt.subplots(figsize=(2, 1))
    ax.set_facecolor(hex_code)
    ax.text(
        0.5, 0.5, hex_code,
        ha='center', va='center',
        color='white' if sum(rgb) < 1.5 else 'black',
        fontsize=14
    )
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

def plot_training_history(history, skip=0):
    history_dict = history.history
    metrics = [m for m in history_dict.keys() if not m.startswith('val_')]
    
    for metric in metrics:
        val_metric = f'val_{metric}'
        plt.figure(figsize=(8, 5))
        train_values = history_dict[metric][skip:]
        epochs_range = range(skip, skip + len(train_values))
        plt.plot(epochs_range, train_values, label=f'Training {metric}', marker='o')
        
        if val_metric in history_dict:
            val_values = history_dict[val_metric][skip:]
            plt.plot(epochs_range, val_values, label=f'Validation {metric}', marker='x')
        
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.title(f'Training and Validation {metric.capitalize()} (skipping first {skip} epochs)')
        plt.legend()
        plt.grid(True)
        plt.show()

def set_clean_style(ax, major_ticks_x=10, minor_ticks_x=0.5,
                    major_ticks_y=10, minor_ticks_y=0.5,
                    minor_tick=False):
    ax.tick_params(axis='both', direction='out', length=3, width=0.4, colors='black', pad=1, which='major')
    ax.tick_params(axis='both', direction='out', length=3, width=0.4, colors='black', pad=8, which='minor')
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(major_ticks_x))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(major_ticks_y))
    
    if minor_tick:
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(minor_ticks_x))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(minor_ticks_y))

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(SMALL_SIZE)
        label.set_fontweight('medium')

    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
    ax.xaxis.labelpad = 2
    ax.yaxis.labelpad = 2
