import pandas as pd
import numpy as np
import talib as ta
from tabulate import tabulate
import logging

logger = logging.getLogger(__name__)

def calculate_technical_success_rate(time_series_data: pd.DataFrame, look_forward_period: int = 10) -> str:
    """
    计算多种技术指标看涨和看跌信号的历史成功率。

    此函数在给定的时间序列数据上回测多种技术指标，
    以确定它们在预测未来价格变动方面的历史成功率。

    Args:
        time_series_data (pandas.DataFrame): 包含至少250天股票数据的DataFrame。
                                             必须包含 'High', 'Low', 'Close', 'Volume' 列。
        look_forward_period (int): 用于判断信号是否成功的未来观察期（交易日数）。

    Returns:
        一个字符串，格式为总结了各指标成功率的表格。
    """
    if len(time_series_data) < 50:  # 确保有足够数据计算50日均线
        logger.warning("数据不足，无法进行完整分析。建议至少提供50天的数据。")
        return "数据不足，无法进行分析。"

    # 确保数据类型正确
    highs = time_series_data['High'].values.astype(float)
    lows = time_series_data['Low'].values.astype(float)
    closes = time_series_data['Close'].values.astype(float)

    results = []
    
    # ==============================================================================
    # --- 1. RSI (相对强弱指数) ---
    # ==============================================================================
    rsi = ta.RSI(closes, 14)
    # 看涨信号: RSI 超卖 (<30)
    bullish_signals = (rsi < 30) & (np.roll(rsi, 1) >= 30)
    bullish_indices = np.where(bullish_signals)[0]
    successful_bullish = 0
    for idx in bullish_indices:
        if idx < len(closes) - look_forward_period:
            if closes[idx + look_forward_period] > closes[idx]:
                successful_bullish += 1
    total_bullish = len(bullish_indices)
    rate = (successful_bullish / total_bullish) * 100 if total_bullish > 0 else 0
    results.append(['RSI 14 Oversold (<30) [上涨]', f"{rate:.2f}%", total_bullish])

    # 看跌信号: RSI 超买 (>70)
    bearish_signals = (rsi > 70) & (np.roll(rsi, 1) <= 70)
    bearish_indices = np.where(bearish_signals)[0]
    successful_bearish = 0
    for idx in bearish_indices:
        if idx < len(closes) - look_forward_period:
            if closes[idx + look_forward_period] < closes[idx]:
                successful_bearish += 1
    total_bearish = len(bearish_indices)
    rate = (successful_bearish / total_bearish) * 100 if total_bearish > 0 else 0
    results.append(['RSI 14 Overbought (>70) [下跌]', f"{rate:.2f}%", total_bearish])

    # ==============================================================================
    # --- 2. SMA Crossover (移动平均线交叉) ---
    # ==============================================================================
    sma20 = ta.SMA(closes, 20)
    sma50 = ta.SMA(closes, 50)
    # 看涨信号: 黄金交叉
    bullish_signals = (sma20 > sma50) & (np.roll(sma20, 1) <= np.roll(sma50, 1))
    bullish_indices = np.where(bullish_signals)[0]
    successful_bullish = 0
    for idx in bullish_indices:
        if idx < len(closes) - look_forward_period:
            if closes[idx + look_forward_period] > closes[idx]:
                successful_bullish += 1
    total_bullish = len(bullish_indices)
    rate = (successful_bullish / total_bullish) * 100 if total_bullish > 0 else 0
    results.append(['SMA 20/50 Golden Cross [上涨]', f"{rate:.2f}%", total_bullish])

    # 看跌信号: 死亡交叉
    bearish_signals = (sma20 < sma50) & (np.roll(sma20, 1) >= np.roll(sma50, 1))
    bearish_indices = np.where(bearish_signals)[0]
    successful_bearish = 0
    for idx in bearish_indices:
        if idx < len(closes) - look_forward_period:
            if closes[idx + look_forward_period] < closes[idx]:
                successful_bearish += 1
    total_bearish = len(bearish_indices)
    rate = (successful_bearish / total_bearish) * 100 if total_bearish > 0 else 0
    results.append(['SMA 20/50 Death Cross [下跌]', f"{rate:.2f}%", total_bearish])

    # ==============================================================================
    # --- 3. MACD (异同移动平均线) ---
    # ==============================================================================
    macd, macd_signal, _ = ta.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
    # 看涨信号: MACD线上穿信号线
    bullish_signals = (macd > macd_signal) & (np.roll(macd, 1) <= np.roll(macd_signal, 1))
    bullish_indices = np.where(bullish_signals)[0]
    successful_bullish = 0
    for idx in bullish_indices:
        if idx < len(closes) - look_forward_period:
            if closes[idx + look_forward_period] > closes[idx]:
                successful_bullish += 1
    total_bullish = len(bullish_indices)
    rate = (successful_bullish / total_bullish) * 100 if total_bullish > 0 else 0
    results.append(['MACD Bullish Crossover [上涨]', f"{rate:.2f}%", total_bullish])
    
    # 看跌信号: MACD线下穿信号线
    bearish_signals = (macd < macd_signal) & (np.roll(macd, 1) >= np.roll(macd_signal, 1))
    bearish_indices = np.where(bearish_signals)[0]
    successful_bearish = 0
    for idx in bearish_indices:
        if idx < len(closes) - look_forward_period:
            if closes[idx + look_forward_period] < closes[idx]:
                successful_bearish += 1
    total_bearish = len(bearish_indices)
    rate = (successful_bearish / total_bearish) * 100 if total_bearish > 0 else 0
    results.append(['MACD Bearish Crossover [下跌]', f"{rate:.2f}%", total_bearish])

    # ==============================================================================
    # --- 4. Bollinger Bands (布林带) ---
    # ==============================================================================
    upper, middle, lower = ta.BBANDS(closes, timeperiod=20)
    # 过滤连续信号的辅助函数
    def get_true_signals(indices):
        true_signals = []
        if len(indices) > 0:
            last_idx = -look_forward_period
            for idx in indices:
                if idx > last_idx + 1:
                    true_signals.append(idx)
                last_idx = idx
        return true_signals

    # 看涨信号: 价格触及下轨
    bullish_indices = get_true_signals(np.where(lows < lower)[0])
    successful_bullish = 0
    for idx in bullish_indices:
        if idx < len(closes) - look_forward_period:
            future_closes = closes[idx + 1 : idx + 1 + look_forward_period]
            future_middles = middle[idx + 1 : idx + 1 + look_forward_period]
            if np.any(future_closes > future_middles):
                successful_bullish += 1
    total_bullish = len(bullish_indices)
    rate = (successful_bullish / total_bullish) * 100 if total_bullish > 0 else 0
    results.append(['Bollinger Lower Band [上涨]', f"{rate:.2f}%", total_bullish])

    # 看跌信号: 价格触及上轨
    bearish_indices = get_true_signals(np.where(highs > upper)[0])
    successful_bearish = 0
    for idx in bearish_indices:
        if idx < len(closes) - look_forward_period:
            future_closes = closes[idx + 1 : idx + 1 + look_forward_period]
            future_middles = middle[idx + 1 : idx + 1 + look_forward_period]
            if np.any(future_closes < future_middles):
                successful_bearish += 1
    total_bearish = len(bearish_indices)
    rate = (successful_bearish / total_bearish) * 100 if total_bearish > 0 else 0
    results.append(['Bollinger Upper Band [下跌]', f"{rate:.2f}%", total_bearish])

    # 格式化最终输出的表格
    headers = ["指标信号", "成功率", "信号总数"]
    return tabulate(results, headers=headers, tablefmt="grid")

# ==============================================================================
# --- 示例用法 ---
# ==============================================================================
if __name__ == '__main__':
    # 创建随机的模拟股票数据用于演示
    print("正在生成模拟股票数据用于演示...")
    data = {}
    # 创建一些趋势使数据更真实
    days = 50
    price_trend = 100 + np.sin(np.linspace(0, 20, days)) * 25 + np.random.randn(days) * 8
    data['Close'] = price_trend
    data['High'] = data['Close'] + np.random.uniform(0, 5, days)
    data['Low'] = data['Close'] - np.random.uniform(0, 5, days)
    data['Volume'] = np.random.uniform(100000, 500000, days)
    
    sample_df = pd.DataFrame(data)

    # 计算并打印成功率表格
    success_table = calculate_technical_success_rate(sample_df, look_forward_period=10)
    print("\n--- 技术指标成功率回测 (包含上涨与下跌信号) ---")
    print(f"分析周期: 最近{days}天")
    print(f"成功标准: 信号出现后10个交易日内的价格表现")
    print(success_table)