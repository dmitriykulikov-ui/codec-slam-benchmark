"""rcParams used for the paper figures (font sizes, spines, PDF font embedding)."""
import matplotlib as mpl

RC = {'axes.labelsize': 8.0, 'axes.linewidth': 0.6, 'axes.spines.right': False, 'axes.spines.top': False, 'axes.titlelocation': 'left', 'axes.titlesize': 8.0, 'font.size': 8.0, 'legend.fontsize': 7.0, 'legend.frameon': False, 'lines.linewidth': 1.2, 'patch.linewidth': 0.6, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.dpi': 300.0, 'xtick.labelsize': 6.0, 'xtick.major.size': 3.0, 'xtick.major.width': 0.6, 'ytick.labelsize': 6.0, 'ytick.major.size': 3.0, 'ytick.major.width': 0.6}

def apply_figure_style():
    mpl.rcParams.update(RC)
