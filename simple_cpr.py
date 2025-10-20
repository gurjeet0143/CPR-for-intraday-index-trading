import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Constants (translated from Pine Script)
AUTO = 'Auto'
DAILY = 'Daily'
WEEKLY = 'Weekly'
MONTHLY = 'Monthly'
QUARTERLY = 'Quarterly'
HALF_YEARLY = 'Half-yearly'
YEARLY = 'Yearly'

TRADITIONAL = 'Traditional'
FIBONACCI = 'Fibonacci'
CLASSIC = 'Classic'
CAMARILLA = 'Camarilla'

# Default input parameters (translated from input.* calls)
def get_defaults():
    return {
        'kind': TRADITIONAL,
        'cpr_time_frame': AUTO,
        'look_back': 1,
        'position_labels': 'Right',
        'line_width': 3,
        'hist_sr_show': True,
        'is_daily_based': True,
        # CPR Customization
        'cpr_show': True,
        'cpr_color': 'blue',
        'cpr_show_lines': True,
        'cpr_show_fill': False,
        'cpr_show_prices': False,
        'cpr_show_labels': False,
        'cpr_line_style': 'Dotted',
        # Pivot Customization
        'sr_show': True,
        'sr_show_labels': True,
        'sr_show_prices': True,
        # Pivot levels show toggles and colors (defaults)
        's0_5_show': False, 's0_5_color': '#FB8C00',
        'r0_5_show': False, 'r0_5_color': '#FB8C00',
        's1_show': True, 's1_color': '#FB8C00',
        'r1_show': True, 'r1_color': '#FB8C00',
        's1_5_show': False, 's1_5_color': '#FB8C00',
        'r1_5_show': False, 'r1_5_color': '#FB8C00',
        's2_show': False, 's2_color': '#FB8C00',
        'r2_show': False, 'r2_color': '#FB8C00',
        's2_5_show': False, 's2_5_color': '#FB8C00',
        'r2_5_show': False, 'r2_5_color': '#FB8C00',
        's3_show': False, 's3_color': '#FB8C00',
        'r3_show': False, 'r3_color': '#FB8C00',
        's3_5_show': False, 's3_5_color': '#FB8C00',
        'r3_5_show': False, 'r3_5_color': '#FB8C00',
        's4_show': False, 's4_color': '#FB8C00',
        'r4_show': False, 'r4_color': '#FB8C00',
        's4_5_show': False, 's4_5_color': '#FB8C00',
        'r4_5_show': False, 'r4_5_color': '#FB8C00',
        's5_show': False, 's5_color': '#FB8C00',
        'r5_show': False, 'r5_color': '#FB8C00',
        # Developing CPR
        'dev_cpr_show': False, 'dev_cpr_color': '#E2BEE7',
        'dev_s1_show': False, 'dev_s1_color': '#E2BEE7',
        'dev_r1_show': False, 'dev_r1_color': '#E2BEE7',
        'extend_dev_cpr_line': False,
        'dev_cpr_show_labels': True, 'dev_cpr_show_prices': False,
        'dev_r1_show_labels': True, 'dev_r1_show_prices': False,
        'dev_s1_show_labels': True, 'dev_s1_show_prices': False,
        'dev_cpr_line_days': 1,
        'check_holidays': True,
        # Fills
        'fill_cpr_r1': False, 'fill_cpr_r1_col': 'rgba(255,165,0,0.29)',
        'fill_cpr_s1': False, 'fill_cpr_s1_col': 'rgba(255,165,0,0.29)',
        # Session Display
        'show_day': False, 'session_hl_line_style': 'Solid',
        'show_day_open': False, 'session_high_color': 'aqua',
        'session_low_color': 'aqua', 'session_open_color': 'aqua',
        'session_open_line_style': 'Dashed',
        'pd_show_labels': True, 'pd_show_prices': True,
    }

# Pivot resolution function (translated from get_pivot_resolution)
def get_pivot_resolution(cpr_time_frame, current_timeframe_minutes):
    if cpr_time_frame == AUTO:
        if current_timeframe_minutes <= 15:
            return 'D'
        else:
            return 'W'
    elif cpr_time_frame == DAILY:
        return 'D'
    elif cpr_time_frame == WEEKLY:
        return 'W'
    elif cpr_time_frame == MONTHLY:
        return 'M'
    elif cpr_time_frame == QUARTERLY:
        return '3M'
    elif cpr_time_frame == HALF_YEARLY:
        return '6M'
    elif cpr_time_frame == YEARLY:
        return '12M'
    return 'M'  # Default

# Pivot calculation functions (translated from traditional, fibonacci, etc.)
def calculate_traditional(prev_high, prev_low, prev_close):
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = pivot * 2 - prev_low
    s1 = pivot * 2 - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    r3 = r1 + prev_high - prev_low
    s3 = s1 - prev_high + prev_low
    r4 = r3 + r2 - r1
    s4 = s3 + s2 - s1
    r5 = r4 + r3 - r2
    s5 = s4 + s3 - s2
    r0_5 = (pivot + r1) / 2
    s0_5 = (pivot + s1) / 2
    r1_5 = (r1 + r2) / 2
    s1_5 = (s1 + s2) / 2
    r2_5 = (r2 + r3) / 2
    s2_5 = (s2 + s3) / 2
    r3_5 = (r3 + r4) / 2
    s3_5 = (s3 + s4) / 2
    r4_5 = (r4 + r5) / 2
    s4_5 = (s4 + s5) / 2
    return {
        'pivot': pivot, 'r0_5': r0_5, 's0_5': s0_5, 'r1': r1, 's1': s1,
        'r1_5': r1_5, 's1_5': s1_5, 'r2': r2, 's2': s2, 'r2_5': r2_5, 's2_5': s2_5,
        'r3': r3, 's3': s3, 'r3_5': r3_5, 's3_5': s3_5, 'r4': r4, 's4': s4,
        'r4_5': r4_5, 's4_5': s4_5, 'r5': r5, 's5': s5
    }

def calculate_fibonacci(prev_high, prev_low, prev_close):
    pivot = (prev_high + prev_low + prev_close) / 3
    pr = prev_high - prev_low
    return {
        'pivot': pivot, 'r1': pivot + 0.382 * pr, 's1': pivot - 0.382 * pr,
        'r2': pivot + 0.618 * pr, 's2': pivot - 0.618 * pr,
        'r3': pivot + pr, 's3': pivot - pr
    }

def calculate_classic(prev_high, prev_low, prev_close):
    pivot = (prev_high + prev_low + prev_close) / 3
    pr = prev_high - prev_low
    r1 = pivot * 2 - prev_low
    s1 = pivot * 2 - prev_high
    r2 = pivot + pr
    s2 = pivot - pr
    r3 = pivot + 2 * pr
    s3 = pivot - 2 * pr
    r4 = pivot + 3 * pr
    s4 = pivot - 3 * pr
    return {
        'pivot': pivot, 'r1': r1, 's1': s1, 'r2': r2, 's2': s2,
        'r3': r3, 's3': s3, 'r4': r4, 's4': s4
    }

def calculate_camarilla(prev_high, prev_low, prev_close):
    pivot = (prev_high + prev_low + prev_close) / 3
    pr = prev_high - prev_low
    r1 = prev_close + pr * 1.1 / 12
    s1 = prev_close - pr * 1.1 / 12
    r2 = prev_close + pr * 1.1 / 6
    s2 = prev_close - pr * 1.1 / 6
    r3 = prev_close + pr * 1.1 / 4
    s3 = prev_close - pr * 1.1 / 4
    r4 = prev_close + pr * 1.1 / 2
    s4 = prev_close - pr * 1.1 / 2
    r5 = (prev_high / prev_low) * prev_close
    s5 = 2 * prev_close - r5
    return {
        'pivot': pivot, 'r1': r1, 's1': s1, 'r2': r2, 's2': s2,
        'r3': r3, 's3': s3, 'r4': r4, 's4': s4, 'r5': r5, 's5': s5
    }

# Main CPR calculation function (translated from calc_pivot and related logic)
def compute_cpr_levels(df, kind=TRADITIONAL, resolution='D', look_back=1, is_daily_based=True):
    """
    Computes CPR and pivot levels.
    df: pandas DataFrame with datetime index, columns ['open', 'high', 'low', 'close']
    resolution: str like 'D', 'W', etc. for resampling.
    look_back: number of most recent pivot periods to return.
    Returns: DataFrame with computed levels.
    """
    if resolution is None:
        resolution = 'D'

    # Resample to the pivot resolution
    ohlc_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    res_df = df.resample(resolution).agg(ohlc_dict)
    res_df = res_df.dropna(how='any')

    # Shift for previous values
    res_df['prev_open'] = res_df['open'].shift(1)
    res_df['prev_high'] = res_df['high'].shift(1)
    res_df['prev_low'] = res_df['low'].shift(1)
    res_df['prev_close'] = res_df['close'].shift(1)

    # Drop rows without previous period
    res_df = res_df.dropna(subset=['prev_high', 'prev_low', 'prev_close'])

    # Compute pivot and CPR core values
    res_df['pivot'] = (res_df['prev_high'] + res_df['prev_low'] + res_df['prev_close']) / 3.0
    res_df['bottom'] = (res_df['prev_high'] + res_df['prev_low']) / 2.0
    res_df['top'] = res_df['pivot'] * 2.0 - res_df['bottom']

    # Ensure bottom <= top
    mask = res_df['bottom'] > res_df['top']
    if mask.any():
        res_df.loc[mask, ['top', 'bottom']] = res_df.loc[mask, ['bottom', 'top']].values

    # Compute levels based on kind
    if kind == TRADITIONAL:
        levels = calculate_traditional(res_df['prev_high'], res_df['prev_low'], res_df['prev_close'])
    elif kind == FIBONACCI:
        levels = calculate_fibonacci(res_df['prev_high'], res_df['prev_low'], res_df['prev_close'])
    elif kind == CLASSIC:
        levels = calculate_classic(res_df['prev_high'], res_df['prev_low'], res_df['prev_close'])
    elif kind == CAMARILLA:
        levels = calculate_camarilla(res_df['prev_high'], res_df['prev_low'], res_df['prev_close'])
    else:
        levels = calculate_traditional(res_df['prev_high'], res_df['prev_low'], res_df['prev_close'])

    # Attach computed series to res_df
    for k, v in levels.items():
        res_df[k] = v

    # Previous Day High/Low (resampled to daily)
    try:
        daily_high = df['high'].resample('D').max().shift(1)
        daily_low = df['low'].resample('D').min().shift(1)
        res_df['pdh'] = daily_high.reindex(res_df.index, method='ffill')
        res_df['pdl'] = daily_low.reindex(res_df.index, method='ffill')
    except Exception:
        res_df['pdh'] = np.nan
        res_df['pdl'] = np.nan

    # Limit to look_back
    if look_back is not None and look_back > 0:
        res_df = res_df.tail(look_back)

    return res_df

# Developing CPR (translated from the last bar logic)
def compute_developing_cpr(current_high, current_low, current_close, kind=TRADITIONAL):
    dpp = (current_high + current_low + current_close) / 3.0
    dbc = (current_high + current_low) / 2.0
    dtc = dpp * 2 - dbc
    dev_top = max(dtc, dbc)
    dev_bot = min(dtc, dbc)
    dev_levels = {'dev_pivot': dpp, 'dev_top': dev_top, 'dev_bot': dev_bot}
    if kind == TRADITIONAL:
        dev_r1 = dpp * 2 - current_low
        dev_s1 = dpp * 2 - current_high
    elif kind == FIBONACCI:
        pr = current_high - current_low
        dev_r1 = dpp + 0.382 * pr
        dev_s1 = dpp - 0.382 * pr
    else:
        dev_r1 = dpp * 2 - current_low  # Default to traditional
        dev_s1 = dpp * 2 - current_high
    dev_levels['dev_r1'] = dev_r1
    dev_levels['dev_s1'] = dev_s1
    return dev_levels

# Example usage and plotting (simplified translation of drawing logic)
def plot_cpr_levels(df, levels_df, developing=False, dev_levels=None, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['close'], label='Close', color='black')

    # Use the last computed pivot period for plotting main levels
    if levels_df.empty:
        ax.set_title('No CPR levels to plot')
        ax.legend()
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path)
            print(f"Saved plot to {save_path}")
        plt.show()
        plt.close(fig)
        return

    last = levels_df.iloc[-1]

    # Plot CPR band
    if 'top' in last and 'bottom' in last and not (np.isnan(last['top']) or np.isnan(last['bottom'])):
        ax.axhspan(last['bottom'], last['top'], alpha=0.15, color='blue', label='CPR Band')

    # Plot pivot and key levels if present
    if 'pivot' in last and not np.isnan(last['pivot']):
        ax.axhline(last['pivot'], color='blue', alpha=0.6, linestyle='-',
                   label='CPR Pivot')
    for lvl in ['r1', 's1', 'r2', 's2', 'r3', 's3', 'r4', 's4']:
        if lvl in last and not np.isnan(last[lvl]):
            color = 'red' if lvl.startswith('r') else 'green'
            ax.axhline(last[lvl], color=color, alpha=0.4, linestyle='--', label=lvl.upper())

    # PDH/PDL
    if 'pdh' in last and not np.isnan(last['pdh']):
        ax.axhline(last['pdh'], color='aqua', linestyle='-.', label='PDH')
    if 'pdl' in last and not np.isnan(last['pdl']):
        ax.axhline(last['pdl'], color='aqua', linestyle='-.', label='PDL')

    # Developing CPR overlays
    if developing and dev_levels:
        if dev_levels.get('dev_top') is not None:
            ax.axhline(dev_levels.get('dev_top'), color='purple', linestyle='--', label='Dev Top')
        if dev_levels.get('dev_pivot') is not None:
            ax.axhline(dev_levels.get('dev_pivot'), color='purple', label='Dev CPR')
        if dev_levels.get('dev_bot') is not None:
            ax.axhline(dev_levels.get('dev_bot'), color='purple', linestyle='--', label='Dev Bot')
        if dev_levels.get('dev_r1') is not None:
            ax.axhline(dev_levels.get('dev_r1'), color='red', linestyle=':', label='Dev R1')
        if dev_levels.get('dev_s1') is not None:
            ax.axhline(dev_levels.get('dev_s1'), color='green', linestyle=':', label='Dev S1')

    ax.set_title('Simple CPR Indicator')
    ax.legend(loc='best', fontsize='small', ncol=2)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"Saved plot to {save_path}")
    plt.show()
    plt.close(fig)

# Sample data generation for testing
def generate_sample_data(start='2025-01-01', periods=100):
    dates = pd.date_range(start=start, periods=periods, freq='1H')
    np.random.seed(42)
    base = 100.0 + np.cumsum(np.random.randn(periods) * 0.5)
    # ensure high >= low and reasonable OHLC behaviour
    highs = base + np.abs(np.random.rand(periods) * 0.8)
    lows = base - np.abs(np.random.rand(periods) * 0.8)
    opens = base + np.random.randn(periods) * 0.2
    closes = base + np.random.randn(periods) * 0.2
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes
    }, index=dates)
    return df

# Run example
if __name__ == "__main__":
    df = generate_sample_data()
    print("Sample OHLC data:")
    print(df.tail())

    # Compute levels (daily pivots, last 1 pivot by default)
    levels_df = compute_cpr_levels(df, kind=TRADITIONAL, resolution='D', look_back=1)

    print("\nComputed CPR Levels:")
    display_cols = [c for c in ['pivot', 'top', 'bottom', 'r1', 's1', 'r2', 's2'] if c in levels_df.columns]
    if not levels_df.empty:
        print(levels_df[display_cols].tail())
    else:
        print("No pivot periods computed (check input data and resolution).")

    # Developing CPR (using last bar)
    last_high, last_low, last_close = df['high'].iloc[-1], df['low'].iloc[-1], df['close'].iloc[-1]
    dev_levels = compute_developing_cpr(last_high, last_low, last_close, kind=TRADITIONAL)
    print("\nDeveloping CPR Levels:")
    for k, v in dev_levels.items():
        print(f"{k}: {v:.2f}")

    # Plot and save result
    plot_cpr_levels(df, levels_df, developing=True, dev_levels=dev_levels, save_path="cpr_plot.png")