# Feature Analysis Report for Stock Trend Prediction

This report summarizes the observations from the scatter plots generated to identify potential relationships between technical indicators and the defined 'Label' (1: close price increases by 1% or more in the next 5 days, 0: otherwise).

## Observations from Scatter Plots:

### 1. RSI vs MACD
- **Observation**: There appears to be a general positive relationship between RSI and MACD values. However, the points for Label 0 and Label 1 are largely intermingled, indicating that these two features alone do not provide a clear separation for predicting the trend.

### 2. RSI vs Bollinger Band Width
- **Observation**: A promising pattern was observed in the "left top" region of the scatter plot, where a concentration of `Label = 1` points was noted, with very few `Label = 0` points. This suggests that a combination of **high RSI** and **low Bollinger Band Width** could be a strong signal for a significant price increase in the next 5 days. This area warrants further investigation as a potential predictive signal.

### 3. MACD vs Bollinger Band Width
- **Observation**: While there was a tendency for more `Label = 1` points in the "top part" of the plot, a clear and significant pattern for distinguishing between Label 0 and Label 1 was not evident. The points remained largely mixed.

### 4. RSI vs Daily Change Percentage
- **Observation**: A significant pattern was identified in the "right bottom" region of the scatter plot, where **only `Label = 1` points** were present, with no `Label = 0` points. This indicates that a combination of **high RSI** and **negative Daily Change Percentage** could be a strong indicator for a significant price increase. This is another highly promising area for predictive signal extraction.

## Conclusion:

Based on the visual analysis of the scatter plots, the combinations of **RSI vs Bollinger Band Width** (specifically the high RSI, low BB Width region) and **RSI vs Daily Change Percentage** (specifically the high RSI, negative Daily Change Percentage region) show the most potential for identifying useful signals to predict stock market trends. These areas suggest that certain market conditions, characterized by these technical indicators, are more likely to lead to a significant price increase.

Further analysis, including statistical methods and machine learning model training, should focus on these identified patterns to confirm their predictive power.