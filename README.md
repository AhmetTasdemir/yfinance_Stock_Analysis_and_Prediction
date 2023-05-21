# yfinance_Stock_Analysis_and_Prediction
In this repo, we will access the stock information of Pepsi company using yfinance API. All materials used are for educational purposes only and are not investment advice
Stock
Stocks, each unit of which is called a share, represent ownership of a company. Stocks owned either directly or through a mutual fund or ETF will likely form the majority of most investors’ portfolios.
Key Definitions
Open: The opening price of a stock at the beginning of a trading day.

High: The highest price a stock reaches during a trading day.

Low: The lowest price a stock reaches during a trading day.

Close: The closing price of a stock at the end of a trading day.

Adj Close: The adjusted closing price of a stock, which takes into account any dividends, stock splits, or other corporate actions that occurred during the day.

Volume: The number of shares of a stock that were traded during a particular trading day. This is an important indicator of market activity and can provide insight into the level of buying and selling interest in a particular stock.


By juxtaposing the daily trade volume(in green) with the daily returns(in blue), it was observed that whenever the volume of shares traded is high, there is a comparatively high rise or fall in the price of the stock leading to the high returns.
Thus, on a given day if an unconventionally high volume of trading takes place, then one can expect a big change in the market in either direction.
The volume of shares traded when coupled with the rise or fall in the Price of stock, in general, is an indicator of the confidence of the traders & investors in a particular company
Correlation Analysis Of Stocks with Pair Plot and Joint Plots
“Never put all your eggs in a single basket”
Whenever we go for the diversification of the portfolio, we would NOT want the stocks to be related to each other. Mathematically, Pearson’s correlation coefficient (also called Pearson’s R value) between any pair of stocks should be close to 0. The idea behind is simple — suppose your portfolio comprises stocks that are highly correlated, then if one stock tumbles, the others might fall too and you’re at the risk of losing all your investment!
I selected the aforementioned stocks to perform the correlation analysis. All these stocks are from different segments of the Industry and Market cap. You are free to choose the stocks of your interest. the procedure remains the same.
# Adj close price of all the stocks
combined_df = yf.download(["PEP","UNH","GM","JNJ"], start="2010-01-01", end="2023-05-12")['Adj Close']
combined_df = combined_df.round(2)
combined_df.head()

Pairplot

Correlation analysis is performed on the daily percentage change(daily returns) of the stock price and not on the stock price.
Jointplot
While the Pair plot provides a visual insight into all possible correlations, the Seaborn joint plot provides detailed information like Pearson’s R-value (Pearson’s correlation coefficient) for each pair of stocks. Pearson’s R-value ranges from -1 to 1. A negative value indicates a negative linear relationship between the variables, while a positive value indicates a positive relationship. Pearson’s R-value closer to 1 (or -1) indicates a strong correlation, while the value closer to 0 indicates a weak correlation.
In addition to Pearson’s R-value, the joint plot also shows the respective histograms on the edges as well as the null hypothesis p-value.


Technical Analysis
Technical Indicators

Investors usually perform due diligence on a handful of companies to select their target companies. There is no guarantee that an investor will make money and some investors lose some, if not all, of their investments hence it is wise not to invest in a company that is going to go bust or that is overvalued and its share price is already too high.
Investors usually perform fundamental analysis on a company to understand whether it is worth buying its stock. Once they have selected the chosen companies to invest their money in, they then need to evaluate when to buy the stock. Time is important in stock investing too. This is where the technical indicators can come in handy.
An investor performs technical analysis to compute technical indicators. These indicators can help an investor determine when to buy or sell a stock.

There are a large number of technical indicators available that are used by the investors. The key is to use a handful of them that meets the trading strategies of the investors and make sense for the current market situation. Too many indicators can clutter the charts and add unnecessary noise.
The technical indicators use the OHLCV data. it means the open, high, low, close, and volume of trades. These measures of a stock can be used to compute technical indicators.
The technical indicators can help us with our investment choices.
There is a large number of technical indicators available. The technical indicators can be grouped into Momentum Indicators, Volume Indicators, Volatility Indicators, Trend Indicators, and Other Indicators.
Crossover Analysis
A crossover analysis is a technical analysis technique used to identify potential buy and sell signals in a stock’s price trend.

To perform a crossover analysis, you will need to plot two moving averages for the stock price, typically a shorter-term moving average (e.g. 50-day moving average) and a longer-term moving average (e.g. 200-day moving average). The crossover occurs when the shorter-term moving average crosses above or below the longer-term moving average, indicating a potential trend reversal.

The rolling function is used to calculate the rolling mean (moving average) of the closing price over a specified window size (50 days and 200 days in this example). The fill_between function is used to highlight the regions where the 50-day moving average is above or below the 200-day moving average.

Crossover analysis is just one of many technical analysis tools used to analyze stock prices, and should not be used in isolation to make investment decisions. It’s important to also consider fundamental analysis, market trends, and other factors that may affect the stock price.

The crossover points are where the 50-day moving average crosses over or under the 200-day moving average. These points can be used as potential buy or sell signals for the stock, depending on the direction of the crossover.

Simple Moving Average Technical Indicator
The simple moving average (SMA) is a widely used technical analysis tool that helps smooth out price action by calculating the average price of a security over a specific period of time, usually 20, 50, or 200 days.
The simple moving average is used by traders to identify trend direction and potential support and resistance levels. When the price is above the SMA, it is considered bullish, and when it is below the SMA, it is considered bearish. In this case, since the price is above the 50-day SMA, it can be considered as bullish.
One important thing to note about SMA is that it gives equal weightage to all the price points in the time period being considered. So, the more recent prices are given the same weightage as the older prices. Therefore, it may not be the most effective tool in volatile markets where prices can fluctuate rapidly.*
The formula for calculating the SMA is as follows:
SMA = (Sum of prices over a given period) / (Number of prices in that period)

Mathematical Intuition
The crossover points are where the two moving averages intersect. These points represent potential buy or sell signals for the stock, depending on the direction of the crossover.
Mathematically, we can calculate the moving averages as follows:
Simple Moving Average = Sum of stock prices over the period / Number of periods

We can then compare the 50-day and 200-day moving averages using the following
Strategy:

If the 50-day moving average > the 200-day moving average, it is a bullish signal
If the 50-day moving average < the 200-day moving average, it is a bearish signal.
The crossover points occur when the two moving averages intersect. At this point, the direction of the crossover will determine whether it is a potential buy or sell signal for the stock.

Exponential Moving Average Technical Indicator
The exponential moving average (EMA) is a type of moving average that gives greater weight to more recent data in the time series, while also taking into account older data. The EMA is calculated using a smoothing factor that places a greater weight on recent data points. This can make it more responsive to price changes compared to a simple moving average (SMA).
As we can see from the plot, the EMA generally tracks the closing price quite closely. We can also see that there are several instances where the buying and selling signals are generated, indicating potential opportunities to enter or exit the market.
It’s important to note that this is just one of many technical indicators that can be used to analyze stock prices, and it should be used in conjunction with other indicators and fundamental analysis to make informed trading decisions.
The formula for calculating the exponential moving average (EMA) is as follows:
EMA = (Close — EMA_prev) * multiplier + EMA_prev

Where:

Close is the current closing price of the asset
EMA_prev is the previous period’s EMA multiplier is a smoothing factor that determines the weight of the current period’s price in the calculation.
The formula for calculating the multiplier is: 2 / (N + 1), where N is the number of periods in the EMA.

After calculating the initial EMA value, we can use the following formula to calculate the EMA for the next period:

EMA[today] = (Price[today] — EMA[yesterday]) * (2 / (1 + N)) + EMA[yesterday]
Strategy:

If the 50-day moving average > the 200-day moving average, it is a bullish signal
If the 50-day moving average < the 200-day moving average, it is a bearish signal.
Observations
This code will plot a graph with the stock price, EMA, buy signals as green triangles, and sell signals as red triangles. The buy and sell signals are determined by comparing the EMA with the stock price.
Some insights that we can gather from this graph include:

The stock price seems to follow the EMA closely, indicating that the EMA is a good indicator of the stock’s trend.
The buy and sell signals can be used to time the trades, as they indicate when the stock price is expected to rise or fall.
There are some false signals in the buy and sell signals, which may result in losses if acted upon.
Therefore, it’s important to use these signals in conjunction with other indicators and perform thorough analysis before making any trading decisions.
MACD Technical Indicator
The Moving Average Convergence Divergence (MACD) indicator is one of the most popular technical oscillator indicators.
It helps us understand the relationship between the moving averages. Convergent is when the lines move closer to each other and divergence is when the lines move away from each other. The lines here are the moving averages.
MACD is a trend-following momentum indicator. It can help us assess the relationship between two moving averages of prices. Subsequently, the MACD indicator can be used to compute a trading strategy that signals us when to buy or sell a stock.
Before I begin, it’s worth mentioning that a moving average is a rolling average value of a predefined historic period. As an instance, the simple 10-day moving average is computed by calculating the average of the past 10 days period. The exponential moving average, on the other hand, assigns higher importance to the recent values. It can help us capture the movements of a stock price better.
Three main steps to calculate MACD:
Step 1: Calculate the MACD line:
Calculate the 26-day exponentially weighted moving average of the price. This is the long-term line.
Calculate the 12-day exponentially weighted moving average of the price. This is the short-term line.
Calculate the difference between the 26-day EMA and 12-day EMA lines. This is the MACD line.
Step 2: Calculate the Signal line from the MACD line:
Calculate the 9 days exponentially weighted moving average of the MACD line. This is known as the signal line.
Step 3: Compute the histogram: Distance between MACD and the Signal
We can then calculate the difference between the MACD and the Signal line and then plot it as a histogram. The histogram can help us find when the cross-over is about to happen.


We can use the cross-over between MACD and the Signal line to create a simple trading strategy. This is where the MACD line and the signal line cross over each other.
Sell Signal: The cross-over: When the MACD line is below the signal line.
Buy Signal: The cross-over: When the MACD line is above the signal line
Bullish vs Bearish:

Bearish: When the MACD and Signal lines are below 0 then the market is bearish.
Bullish: When the MACD and Signal lines are above 0 then the market is bullish.
Key Points

MACD is based on moving averages which imply that the past can impact the future. This is not always true. Additionally, there is a lag present due to the moving averages hence the generated signals are after the move has started.
The standard setting for MACD is the difference between the 12- and 26-period EMAs. We could use MACD(5,35,5) for more sensitive stocks and MACD(12,26,9) might be better suited for weekly charts. It all depends on the investor.
One keynote to remember is to always analyze the short and long-term price trends along with other factors. And remember sometimes a stock that might appear overbought might still move upwards due to other market factors.

RSI Technical Indicator
RSI stands for Relative Strength Index. It’s a widely used technical indicator and this is mainly due to its simplicity. It relies on the market and we can use the indicator to determine when to buy or sell a stock.
RSI requires us to compute the recent gains and losses. The recent specified time period is subjective in nature. We use the RSI indicator to measure the speed and change of price movements.
RSI is an oscillating indicator. It can help us understand the momentum better. Note, momentum is the change of price and size. Therefore, the RSI indicator can help us understand when the stock price will change its trend.
The key to using this indicator is to understand whether a stock is overbought or oversold.
Calculation:

The calculation is extremely simple.
Firstly, we have to determine the time period. Usually, a 14 day time period is chosen but it could depend on the investor’s own view of the market and the stock.
Secondly, we have to compute the relative strength which is known as RS. RS is the average gain over the average loss. To explain it further, RS is the average gain when the price was moving up over the average loss when the price change was negative.
Calculate RSI as 100 — (100/(1+RS))
The RSI value is between 0–100
Strategy:

Overbought: When the RSI is above 70%. Essentially, overbought is when the price of a stock has increased quickly over a small period of time, implying that it is overbought.
The price of an overbought stock usually decreases in price.
Oversold: When the RSI is below 30%. Essentially, oversold is when the price of a stock has decreased quickly over a small period of time, implying that it is oversold. The price of an oversold stock usually increases in price.
There are way too many strategies that are dependent on the RSI indicator.
A simple strategy is to use the RSI such that:
Sell: When RSI increases above 70%
Buy: When RSI decreases below 30%.
We might decide to use different parameters. The point is that we can optimize the parameters that meet our trading style, the market, and the stock we are interested in.
Key Points

The signals are not always accurate. The RSI signals are dependent on the price of the stock only and this is not the only factor that can change the price of a stock. Plus it’s highly subjective.
For instance, a company can launch a product when a stock is oversold and that could further increase the price of the stock.
Therefore, always consider the market factors and also use the short and long-term price trends when buying or selling a stock.

Bollinger Bands Technical Indicator
It is one of the most popular technical indicators. And this is mainly due to its simplicity.
There are two main components of a Bollinder band indicator:
Volatility Bolinger Bands
Moving averages
Essentially, the steps are:
Middle band: Calculate the moving average of the price, usually 20 days moving average.
Upper band: Calculate two standard deviations above the moving average.
Lower band: Calculate two standard deviations below the moving average.
The more volatile the stock prices, the wider the bands from the moving average. It’s important to look at the shape/trend of the bands along with the gap between them to understand the trend and stock better.
The standard deviations capture the volatile movements and hence we compute standard deviations above and below the upper and lower bands to capture the outliers. Consequently, 95% of the price movements will fall between the two standard deviations
Strategy:
A simple trading strategy could be to:
Sell: As soon as the market price touches the upper Bollinger band
Buy: As soon as the market price touches the lower Bollinger band
This is based on the assumption that the stock must fall back (from the uptrend) and eventually touch the bottom band.
At times, the Bollinger Band Indicator signals us to buy a stock but an external market event such as negative news can change the price of the stock. Therefore it’s important to use the indicator as just an indicator that can sometimes be wrong.

The library has a bonus function. We can add all of the available technical indicators that have been coded in the ta library by calling the add_all_ta_features function.
For this code to work, create a data frame and ensure it contains the Open, High, Low, and Close columns.
Conclusion

The signals generated by the technical indicators are theoretical in nature. * There is no guarantee that the signals are going to be absolutely applicable all the time. Plus the market can behave in an unexpected manner.
Investors can lose some, if not all, of their investments therefore the indicators should be used wisely. It’s important to note that stock investment should not be taken lightly. On top of that, the market can not always be timed.
Usually, the technical indicators are combined together to achieve a better indicator. As an instance, the Bollinger band and MACD indicators can be combined with the RSI measure to better decide whether it is the right time to buy/sell.
It is important to adjust the parameters and decide the optimum trading strategy based on your view of the market, your trading style, and the investment stock. Always backtest the trading strategy.

Introduction to ARIMA Models
ARIMA models are a type of statistical model that can be used to analyse and forecast time series data. It gives a simple yet powerful way for creating time series forecasts by explicitly catering to a set of common structures in time series data.

ARIMA is an acronym for AutoRegressive Integrated Moving Average. It’s a more complex version of the AutoRegressive Moving Average, with the addition of integration.

An ARIMA model is characterized by 3 terms: p, d, q where,

p is the order of the AR term. The number of lag observations included in the model, also called the lag order.
q is the size of the moving average window, also called the order of moving average.
d is the number of differencing required to make the time series stationary.
What does ARIMA(p, d, q) mean?

For example :

ARIMA(1, 0, 3) signifies that you’re combining a 1st order Auto-Regressive model and a 3rd order Moving Average model to describe some response variable (Y) in your model. It’s a good idea to think about it this way: (AR, I, MA). In simple words, this gives your model the following appearance:
Y = (Auto-Regressive Parameters) + (Moving Average Parameters)
The 0 between the 1 and the 3 represents the ‘I’ part of the model (the Integrative component), which denotes a model that takes the difference between response variable data — this can be done with non-stationary data, but you don’t appear to be dealing with that, so ignore it.
ARIMA(2, 1, 2) signifies that you’re combining a 2nd order AR model and also a 2nd order MA model to describe Y. d = 1st denotes that the model used 1 order differencing to make the data stationary.
Just like these examples we have to find perfect order of p, d and q to fit the best model.

There are a number of ways to find values of p, q and d:

look at an autocorrelation graph of the data (will help if Moving Average (MA) model is appropriate)
look at a partial autocorrelation graph of the data (will help if AutoRegressive (AR) model is appropriate)
look at extended autocorrelation chart of the data (will help if a combination of AR and MA are needed)
try Akaike’s Information Criterion (AIC) on a set of models and investigate the models with the lowest AIC values
try the Schwartz Bayesian Information Criterion (BIC) and investigate the models with the lowest BIC values
Before working with non-stationary data, the Autoregressive Integrated Moving Average (ARIMA) Model converts it to stationary data. One of the most widely used models for predicting linear time series data is this one.

The ARIMA model has been widely utilized in banking and economics since it is recognized to be reliable, efficient, and capable of predicting short-term share market movements.

Problem Statement: In this notebook, we are going to use the ARIMA, SARIMA, and Auto ARIMA models to forecast the stock price of Apple.


A time series is also thought to include three systematic components: level, trend, and seasonality, as well as one non-systematic component termed noise.

The components’ definitions are as follows:

The level is the sum of all the values in a series.
The trend is the upward or downward movement of the series’ value.
The series’ short-term cycle is known as seasonality.
Noise is the term for the random variation in the series.
Check for stationarity
Time series analysis only works with stationary data, so we must first determine whether a series is stationary. Stationary time series is when the mean and variance are constant over time. It is easier to predict when the series is stationary.

What does it mean for data to be stationary?

The mean of the series should not be a function of time. Because the mean increases over time, the red graph below is not stationary.

The variance of the series should not be a function of time. Homoscedasticity is the term for this characteristic. The varying spread of data over time can be seen in the red graph.

Finally, neither the I th term nor the (I + m) th term’s covariance should be a function of time. As you can see in the graph below, the spread gets less as time goes on. As a result, the’red series’ covariance does not remain constant throughout time.

ADF (Augmented Dickey-Fuller) Test
The Dickey-Fuller test is one of the most extensively used statistical tests. It can be used to establish whether a series has a unit root and, as a result, whether the series is stationary. The null and alternate hypotheses for this test are: Distinguish between point to point links and multi point links Null Hypothesis: The series has a unit root (a =1).

Alternative Hypothesis: There is no unit root in the series.

The series is considered to be non-stationary if the null hypothesis is not rejected. As a result, the series can be linear or difference stationary. If both the mean and standard deviation are flat lines, the series becomes stationary (constant mean and constant variance).


We can’t reject the Null hypothesis because the p-value is bigger than 0.05. Furthermore, the test statistics exceed the critical values. As a result, the data is not stationary.

Differencing is a method of transforming a non-stationary time series into a stationary one. This is an important step in preparing data to be used in an ARIMA model. So, to make the data stationary, we need to take the first-order difference of the data. Which is just another way of saying, subtract today’s close price from yesterday’s close price.

Do differencing until it converts into stationary data where mean and variance are constant


The p-value is obtained is less than significance level of 0.05 and the ADF statistic is lower than any of the critical values.

We can reject the null hypothesis. So, the time series is in fact stationary.

Decompose the time series : To start with, we want to decompose the data to seperate the seasonality, trend and residual. Since we have 3 years of stock data. We would expect there’s a yearly or weekly pattern. Let’s use a function seasonal_decompose in statsmodels to help us find it.

Check the Trend, Seasonality and Residual
Trend — general movement over time
Seasonal — behaviours captured in individual seasonal periods
Residual — everything not captured by trend and seasonal components
Additive vs. multiplicative time series components

There are two techniques for combining time series components:
Additive

The term additive means individual components (trend, seasonality, and residual) are added together:
𝑦𝑡=𝑇𝑡+𝑆𝑡+𝑅𝑡
An additive trend indicates a linear trend, and an additive seasonality indicates the same frequency (width) and amplitude (height) of seasonal cycles
Multiplicative

The term multiplicative means individual components (trend, seasonality, and residuals) are multiplied together:
𝑦𝑡=𝑇𝑡+𝑆𝑡+𝑅𝑡
A multiplicative trend indicates a non-linear trend (curved trend line), and a multiplicative seasonality indicates increasing/decreasing frequency (width) and/or amplitude (height) of seasonal cycles
Both trend and seasonality can be additive or multiplicative, which means there are four ways these can be combined:

Additive trend and additive seasonality
Additive trend means the trend is linear (straight line), and additive seasonality means there aren’t any changes to widths or heights of seasonal periods over time
Additive trend and multiplicative seasonality
Additive trend means the trend is linear (straight line), and multiplicative seasonality means there are changes to widths or heights of seasonal periods over time
Multiplicative trend and additive seasonality
Multiplicative trend means the trend is not linear (curved line), and additive seasonality means there aren’t any changes to widths or heights of seasonal periods over time
Multiplicative trend and multiplicative seasonality
Multiplicative trend means the trend is not linear (curved line), and multiplicative seasonality means there are changes to widths or heights of seasonal periods over time
The seasonal_decompose() function from statsmodels excepts at least two parameters:

x: array — your time series.
model: str — type of seasonal component, can be either additive or multiplicative. The default value is additive.
Additive Model

Multiplicative Model

Now we’ll create an ARIMA model and train it using the train data’s stock closing price. So, let’s visualize the data by dividing it into training and test sets


Auto Correlation and Partial Auto Correlation
From correlation to autocorrelation

Both terms are tightly connected. Correlation measures the strength of the linear relationship between two sequences:
The closer the correlation to +1, the stronger the positive linear relationship
The closer the correlation to -1, the stronger the negative linear relationship
The closer the correlation to 0, the weaker the linear relationship
Autocorrelation is the same, but with a twist — you’ll calculate a correlation between a sequence with itself lagged by some number of time units
Before calculating autocorrelation, you should make the time series stationary (the mean, variance, and covariance shouldn’t change over time)
Auto-correlations

After a time series has been stationarized by differencing, the next step in fitting an ARIMA model is to determine whether AR or MA terms are needed to correct any autocorrelation that remains in the differenced series.

By looking at the autocorrelation function (ACF) and partial autocorrelation (PACF) plots of the differenced series, you can tentatively identify the numbers of AR and/or MA terms that are needed.

Partial autocorrelation

This one is a bit tougher to understand. It does the same as regular autocorrelation — shows the correlation of a sequence with itself lagged by some number of time units. But there’s a twist. Only the direct effect is shown, and all intermediary effects are removed.
For example, you want to know the direct relationship between the stock price today and 12 months ago. You don’t care about anything in between
Autocorrelation function plot (ACF): Autocorrelation refers to how correlated a time series is with its past values whereas the ACF is the plot used to see the correlation between the points, up to and including the lag unit. In ACF, the correlation coefficient is in the x-axis whereas the number of lags is shown in the y-axis.
Normally, we employ either the AR term or the MA term in an ARIMA model. Both of these phrases are rarely used on rare occasions. The ACF plot is used to determine which of these terms we should utilise for our time series.

If the autocorrelation at lag 1 is positive, we utilise the AR model.
If the autocorrelation at lag 1 is negative, we employ the MA model.
We move on to Partial Autocorrelation function plots (PACF) after plotting the ACF plot.

Partial Autocorrelation function plots (PACF) A partial autocorrelation is a summary of the relationship between an observation in a time series with observations at prior time steps with the relationships of intervening observations removed. The partial autocorrelation at lag k is the correlation that results after removing the effect of any correlations due to the terms at shorter lags.
If the PACF plot drops off at lag n, then use an AR(n) model and if the drop in PACF is more gradual then we use the MA term.

Use AR terms in the model when the

ACF plots show autocorrelation decaying towards zero
PACF plot cuts off quickly towards zero
ACF of a stationary series shows positive at lag-1
Use MA terms in the model when the model is

Negatively Autocorrelated at Lag — 1
ACF that drops sharply after a few lags
PACF decreases more gradually
How to interpret ACF and PACF plots

Time series models you’ll soon learn about, such as Auto Regression (AR), Moving Averages (MA), or their combinations (ARMA), require you to specify one or more parameters. These can be obtained by looking at ACF and PACF plots.
In a nutshell:
If the ACF plot declines gradually and the PACF drops instantly, use Auto Regressive model.
If the ACF plot drops instantly and the PACF declines gradually, use Moving Average model.
If both ACF and PACF decline gradually, combine Auto Regressive and Moving Average models (ARMA).
If both ACF and PACF drop instantly (no significant lags), it’s likely you won’t be able to model the time series.

To estimate the amount of AR terms(p), you need to look at the PACF plot. First, ignore the value at lag 0. It will always show a perfect correlation, since we are estimating the correlation between today’s value with itself. Note that there is a coloured area in the plot, representing the confidence interval. To estimate how much AR terms you should use, start counting how many spikes are above or below the confidence interval before the next one enter the coloured area. So, looking at the PACF plot above, we can estimate to use 0 AR terms for our model, since no any spikes are out of the confidence interval.
To calculate d, all you need to know how many differencing was used to make the series stationary. In our case, we have used order of 1st order differencing to make our data stationary.
To estimate the amount of MA terms (q), this time you will look at ACF plot. The same logic is applied here: how many spikes are above or below the confidence interval before the next spike enters the coloured area? Here, we can estimate 0 MA terms, since no spike is out of the confidence interval.
So, we will use (0,1,0) order to fit ARIMA model.

We can also use different orders of p, d and q to get the best order with lowest AIC.


Although our model is on average side but this model has trouble forecasting long-term data. This is possible because ARIMA is a sensitive algorithm and not a broad algorithm for predicting. Stock data, on the other hand, rarely show seasonality that can be detected using the ARIMA model. Forecasting is thought to be easier if there is a visible or hidden pattern that repeats itself throughout time. Stock prices, on the other hand, are far too complicated to be modelled. There are so may external and dynamic factor affecting the price.

**A problem with ARIMA is that it does not support seasonal data. That is a time series with a repeating cycle. ARIMA expects data that is either not seasonal or has the seasonal component removed, e.g. seasonally adjusted via methods such as seasonal differencing.**

Introduction to SARIMA Models
SARIMA (Seasonal ARIMA) is a modification of ARIMA that explicitly allows univariate time series data with a seasonal component. SARIMA accepts an additional set of parameters (P,D,Q)m that specify the model’s seasonal components.

P: Seasonal auto regressive order
D: Seasonal difference order
Q: Seasonal moving average order
m: The number of time steps for a single seasonal period
This is written as (p,d,q)×(P,D,Q)m.

From the ACF and PACF that we have plotted, we can determine the value of Seasonal (P,D,Q). In ACF and PACF, we have one spike at lag 3 that is out of confidence interval and also there is no significant correlation at lag 3 and lag 6. So, the order of P and Q is (1, 1). As we have used differencing of 1 to make data stationary so, D = 1. So, the best order for SARIMA is(0,1,0)x(1,1,1)3

Auto ARIMA
Automatically discover the optimal order for an ARIMA model. After identifying the most optimal parameters for an ARIMA model, the auto arima function provides a fitted ARIMA model. This function is based on the commonly used forecast::auto. Arima R function.

The auro arima function fits models within the start p, max p, start q, max q ranges using differencing tests (e.g., Kwiatkowski–Phillips–Schmidt–Shin, Augmented Dickey–Fuller, or Phillips–Perron) to identify the order of differencing, d. If the seasonal option is enabled, D, auto arima additionally aims to identify the ideal P and Q hyper-parameters after doing the Canova-Hansen to determine the optimal order of seasonal differencing.


Top left: The residual errors appear to have a uniform variance and fluctuate around a mean of zero.

Top Right: The density plot on the top right suggests a normal distribution with a mean of zero.

Bottom left: The red line should be perfectly aligned with all of the dots. Any significant deviations would indicate a skewed distribution.

Bottom Right: The residual errors are not autocorrelated, as shown by the Correlogram, also known as the ACF plot. Any autocorrelation would imply that the residual errors have a pattern that isn’t explained by the model. As a result, you’ll need to add more Xs (predictors) to the model.


