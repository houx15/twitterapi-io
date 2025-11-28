"""
Plot comparison charts for Weibo and Twitter daily opinion metrics.

This script generates three PDF figures comparing avg_opinion, weighted_opinion,
and user_avg_opinion between Weibo and Twitter data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime

from config import SENTIMENT_OUTPUT_DIR

# Set font to serif (will use available serif font like DejaVu Serif)
# plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

# Color definitions
weibo_color = "#ff7333"
twitter_color = "#20AEE6"

# Line style settings
line_style = "solid"
linewidth = 5
alpha = 0.7

# Output directory
FIGURES_DIR = Path("./figures")
FIGURES_DIR.mkdir(exist_ok=True)

# Sliding window settings
WINDOW_SIZE = 3  # Number of days for moving average (can be adjusted)


def apply_sliding_window(df, metric_name, window_size=WINDOW_SIZE):
    """
    Apply sliding window (moving average) to smooth data.
    
    Args:
        df: DataFrame with date and metric columns
        metric_name: Name of the metric column to smooth
        window_size: Size of the sliding window (number of days)
    
    Returns:
        DataFrame with smoothed metric (original metric + smoothed version)
    """
    df = df.copy()
    df = df.sort_values("date")
    
    # Apply rolling mean (moving average)
    smoothed_col = f"{metric_name}_smoothed"
    df[smoothed_col] = df[metric_name].rolling(window=window_size, center=True, min_periods=1).mean()
    
    return df


def load_data():
    """Load Weibo and Twitter daily opinion data."""
    weibo_path = Path(SENTIMENT_OUTPUT_DIR) / "weibo_daily_opinion.parquet"
    twitter_path = Path(SENTIMENT_OUTPUT_DIR) / "twitter_daily_opinion.parquet"
    
    if not weibo_path.exists():
        raise FileNotFoundError(f"Weibo data not found: {weibo_path}")
    if not twitter_path.exists():
        raise FileNotFoundError(f"Twitter data not found: {twitter_path}")
    
    weibo_df = pd.read_parquet(weibo_path)
    twitter_df = pd.read_parquet(twitter_path)
    
    # Convert date to datetime
    weibo_df["date"] = pd.to_datetime(weibo_df["date"])
    twitter_df["date"] = pd.to_datetime(twitter_df["date"])
    
    # Sort by date
    weibo_df = weibo_df.sort_values("date")
    twitter_df = twitter_df.sort_values("date")
    
    return weibo_df, twitter_df


def plot_metric(ax, weibo_df, twitter_df, metric_name, ylabel, use_smoothing=True, window_size=WINDOW_SIZE):
    """
    Plot a single metric comparison.
    
    Args:
        ax: Matplotlib axes object
        weibo_df: Weibo DataFrame
        twitter_df: Twitter DataFrame
        metric_name: Name of the metric column
        ylabel: Y-axis label
        use_smoothing: Whether to apply sliding window smoothing
        window_size: Size of sliding window for smoothing
    """
    # Apply sliding window smoothing if requested
    if use_smoothing:
        weibo_df = apply_sliding_window(weibo_df, metric_name, window_size)
        twitter_df = apply_sliding_window(twitter_df, metric_name, window_size)
        smoothed_col = f"{metric_name}_smoothed"
        weibo_values = weibo_df[smoothed_col]
        twitter_values = twitter_df[smoothed_col]
    else:
        weibo_values = weibo_df[metric_name]
        twitter_values = twitter_df[metric_name]
    
    # Plot lines
    ax.plot(
        weibo_df["date"],
        weibo_values,
        color=weibo_color,
        linestyle=line_style,
        linewidth=linewidth,
        alpha=alpha,
        label="Weibo, China"
    )
    
    ax.plot(
        twitter_df["date"],
        twitter_values,
        color=twitter_color,
        linestyle=line_style,
        linewidth=linewidth,
        alpha=alpha,
        label="Twitter, USA"
    )

    x_min = min(weibo_df["date"].min(), twitter_df["date"].min())
    x_max = max(weibo_df["date"].max(), twitter_df["date"].max())
    x_min = x_min - (x_max - x_min) * 0.03
    
    if metric_name == "weighted_opinion":
        # Add y=0 horizontal line (dashed, grey, width 2)
        ax.axhline(y=0, color='grey', linestyle='--', linewidth=2, zorder=0)
        
        # Add "neutral" annotation on the left at y=0
        ax.text(
            x_min, 0.05,
            "neutral",
            fontsize=10,
            color='grey',
            verticalalignment='center',
            horizontalalignment='left'
        )
    
    # Get y-axis limits for top/bottom annotations
    # Use the actual plotted values (smoothed or original) for consistency
    all_values = pd.concat([weibo_values, twitter_values])
    y_max = all_values.max()
    y_min = all_values.min()
    y_range = y_max - y_min
    
    # Add padding to y-axis for better visualization
    # This ensures consistent visual appearance across different metrics
    y_padding = max(0.1 * y_range, 0.1)  # At least 10% padding or 0.1 unit
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    y_top = y_max + 0.05 * y_range  # Near top
    y_bottom = y_min - 0.05 * y_range  # Near bottom

    # Add "support AI benefits" annotation at top
    
    ax.text(
        x_min, y_top,
        "AI benefits",
        fontsize=10,
        color='black',
        verticalalignment='top',
        horizontalalignment='left'
    )
    
    # Add "concerns AI harms" annotation at bottom
    ax.text(
        x_min, y_bottom,
        "AI concerns",
        fontsize=10,
        color='black',
        verticalalignment='bottom',
        horizontalalignment='left'
    )
    
    # Set labels
    ax.set_xlabel("Time", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    
    # Format x-axis to show only months with 45-degree rotation
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add legend
    ax.legend(loc='lower right', fontsize=10)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')


def main(use_smoothing=True, window_size=WINDOW_SIZE):
    """
    Main function to generate all plots.
    
    Args:
        use_smoothing: Whether to apply sliding window smoothing (default: True)
        window_size: Size of sliding window for smoothing in days (default: 3)
    """
    print("Loading data...")
    weibo_df, twitter_df = load_data()
    
    print(f"Weibo data: {len(weibo_df)} dates")
    print(f"Twitter data: {len(twitter_df)} dates")
    
    if use_smoothing:
        print(f"Applying sliding window smoothing with window size: {window_size} days")
    
    # Get today's date for filename
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    # Create individual plots
    metrics = [
        ("avg_opinion", "Average Opinion"),
        ("weighted_opinion", "LikeCount Weighted Opinion"),
        ("user_avg_opinion", "User-level Average Opinion")
    ]
    
    for metric_name, ylabel in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_metric(
            ax, 
            weibo_df, 
            twitter_df, 
            metric_name, 
            ylabel,
            use_smoothing=use_smoothing,
            window_size=window_size
        )
        # Use subplots_adjust to ensure consistent plot area size across all plots
        # This ensures all plots have the same actual plotting area regardless of label lengths
        plt.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.15)
        
        output_path = FIGURES_DIR / f"{metric_name}_comparison_{today_str}.pdf"
        fig.savefig(output_path, format="pdf", bbox_inches='tight')
        print(f"Individual plot saved to: {output_path}")
        plt.close(fig)
    
    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()

