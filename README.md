Consumer Sentiment as a Leading Indicator: Reddit NLP Signal for Restaurant Chain Earnings Prediction

OVERVIEW

This project investigates whether Reddit consumer sentiment can serve as a leading indicator for earnings surprises across major restaurant chain stocks. Using historical posts scraped from company-specific subreddits, I built an end-to-end alternative data pipeline that scores sentiment using VADER NLP, constructs a weekly time series signal, and backtests its predictive accuracy against quarterly earnings outcomes.


HYPOTHESIS

Consumer discussions on Reddit reflect real-time shifts in company perception that manifest in same-store sales performance and ultimately drive earnings outcomes. If sentiment deteriorates meaningfully in the 30 days before an earnings announcement, it may signal an increased probability of a miss.


PIPELINE

1. I scraped 12,092 historical Reddit posts across 5 subreddits using the Arctic Shift API
2. Pulled stock prices and earnings dates via yfinance
3. Scored sentiment using VADER
4. Aggregated the data into a weekly time series
5. Normalized by ticker using z-scores
6. I fed the pre-earnings sentiment signal into a logistic regression model



COMPANY SELECTION: CMG, MCD, DPZ, WEN, SHAK

1. My original analysis covered SBUX, NKE, and CMG. I ended up dropping and replacing NKE and SBUX.
2. The r/starbucks thread is full of employees talking about working conditions and unionization, rather than customer experience sentiment. The model actually confirmed this, as SBUX had high positive sentiment before misses, which is the opposite of what should happen if the signal worked.
3. In the r/Nike thread, there were a bunch of posts where people were reselling things or talking about new releases. It wasn't really representative of Nike's broader consumer base. There was also only one earnings miss in the whole span of the data, so I couldn't really train a meaningful classifier with that.
4. After removing those two companies, I added a few other restaurant chains where the subreddits were more clearly customer-focused (people complaining about portion sizes, wait times, new menu items, etc.), as these are better indicators of sales.



METHODOLOGY

1. I used VADER for sentiment scoring after testing FinBERT first. Despite FinBERT being finance-specific, VADER performed better on this dataset — Reddit’s informal language aligns better with VADER’s social media training.
2. Weighted posts by log engagement score, so viral posts carry more signal than obscure ones.
3. Normalized sentiment by ticker using z-scores rather than min-max scaling. Each subreddit has a different baseline tone, so raw scores aren’t comparable across companies. Z-scoring makes a -1.5 score mean the same thing for every ticker.
4. I used a 30-day pre-earnings window, which is long enough to capture a trend, but short enough to stay relevant to the quarter.
5. I applied class_weight=‘balanced’ to the logistic regression because 82% of events were beats. Without it the model just predicts a beat every time.


    
SIGNAL CONSTRUCTION
1. Aggregated to weekly sentiment scores by ticker
2. Applied a 4-week rolling average to smooth noise from individual viral posts
3. I decided to use z-scores by ticker so that scores are comparable across companies, because the threads for each company had varying baseline tones, so it wasn't apples to apples. A -1.5 z-score means the same thing for both after normalizing. I considered min-max scaling first, but z-score felt more appropriate because I didn't want one extreme data point to become the floor or ceiling and distort everything. Z-scoring is more stable for detecting deviations from a company's normal baseline.
4. Calculated average z-scored sentiment in the 30-day window preceding each earnings announcement



MODEL
1. Supervised binary classification using logistic regression
2. Predicts earnings beat/miss outcome from pre-earnings sentiment score
3. Applied class_weight='balanced' to address 82/18 beat/miss class imbalance. I priorited recall on misses over raw accuracy.
4. To avoid overfitting, I used 5-fold cross-validation.




RESULTS
1. 12,092 posts scraped across 5 tickers (January 2022 — June 2024)
2. 50 earnings events, 10 per ticker
3. Beat rate: 82%
4. Cross-validated accuracy: 78% (+/- 7.5%)
5. Baseline: 82%
6. Miss recall: 78% — correctly identified 7 of 9 earnings misses
7. Beat precision: 0.94 — (when the model predicts a beat it’s accurate 94% of the time)
8. Miss mean pre-earnings sentiment: -0.83 vs Beat mean: 0.02 — meaningful separation between the two distributions




DATA VISUALIZATIONS

Chart 1: Reddit sentiment z-score time series vs earnings outcomes for all 5 tickers — green solid lines = beat, red dashed lines = miss

![Sentiment Time Series](chart1_sentiment_timeseries.png)


Chart 2: Distribution of 30-day pre-earnings sentiment scores — beats vs misses

![Sentiment Distribution](chart2_sentiment_distribution.png)


Chart 3: Confusion matrix — balanced logistic regression predictions

![Confusion Matrix](chart3_confusion_matrix.png)



INVESTMENT CONCLUSION

Reddit consumer sentiment contains a detectable signal ahead of earnings surprises for restaurant chain stocks. Before earnings misses, pre-earnings sentiment averaged -0.83 standard deviations below each company’s own baseline. Before beats, it averaged just +0.02 — essentially neutral. This implies that positive sentiment doesn’t reliably predict a beat, but unusually negative sentiment does appear to precede misses. The signal serves as a warning indicator, rather than a two-sided predictor. Most importantly, this deterioration was detectable 30 days before earnings announcements, suggesting Reddit captures shifts in consumer perception before they show up in reported results.

The model correctly identified 7 of 9 earnings misses using only sentiment, achieving 78% miss recall after accounting for class imbalance. In a trading context, catching misses matters more than raw accuracy, so optimizing for recall over accuracy was the right tradeoff here.

One of the more interesting findings was that subreddit composition matters as much as subreddit size. r/starbucks failed as a signal source because its subreddit is dominated by employees discussing labor conditions rather than customers discussing store experience. The best alternative data subreddits are ones where the dominant voice is a customer, as their experience feedback is reflected in future sales.

In practice, I would use this signal as a pre-earnings screening tool — flagging names where sentiment has dropped more than 1.5 standard deviations below their own baseline in the 30-day window before earnings as elevated miss risk candidates. This threshold would need to be backtested to find the optimal value, but -1.5 is a reasonable starting point given that the average pre-earnings sentiment before observed misses was -0.83 standard deviations.

9 misses across 50 earnings events constitutes exploratory research for a signal. The framework is sound but the data needs to scale. A production version would require 3-5 years of data across a broader universe of tickers and backtesting of the 1.5 standard deviation threshold against alternatives. Additionally, I only scored post titles and body text — comment-level sentiment was not captured. Reddit comments often contain more detailed consumer feedback than the post itself, and may represent an additional signal worth exploring.



LIMITATIONS

1. Sample size of 50 earnings events and 9 misses is exploratory rather than production-grade
2. A robust signal would require 3-5 years of data across a broader ticker universe
3. Subreddit engagement varies significantly by company — SHAK has far fewer posts than CMG or MCD
4. Reddit users are not representative of the full consumer population
5. The optimal sentiment threshold for flagging increased miss risk would need to be backtested across multiple threshold values. I assumed a 1.5 standard deviation as a starting point.
6. Only post titles and body text were scored, comment-level sentiment is not captured.



NEXT STEPS

1. Expand ticker list and date ranges to increase sample size
2. Combine Reddit sentiment with transaction-level alternative data (maybe credit card or foot traffic data) for a multi-signal model.
3. I could test the signal decay at varying pre-earnings windows
4. Explore ticker-specific models rather than a pooled cross-ticker approach
5. Test additional threshold values (1.0, 1.5, 2.0 standard deviations) to optimize the miss-flagging benchmark.
