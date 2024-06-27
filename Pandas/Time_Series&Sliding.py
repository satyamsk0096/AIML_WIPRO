
#Time Series Analysis
#Working with date and time data
import numpy as np
import pandas as pd

# Creating a sample DataFrame with date range
dates = pd.date_range('20230101', periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
print("DataFrame with dates:\n", df)

#Resampling and frequency conversion
# Resampling to a different frequency
resampled = df.resample('M').mean()
print("Resampled DataFrame (monthly mean):\n", resampled)

#Rolling and expanding windows
# Applying rolling window
rolling = df.rolling(window=2).mean()
print("Rolling window (mean):\n", rolling)

# Applying expanding window
expanding = df.expanding(min_periods=1).mean()
print("Expanding window (mean):\n", expanding)