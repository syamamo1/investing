import matplotlib.pyplot as plt
from matplotlib import dates as mdates
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import os
import datetime

from utils import Timers
from html_generator import make_nice_string

# Calculate bollinger bands
def get_bollinger_bands(df, window=21, std=2):
    '''
    - Bollinger Bands is a technical analysis tool used to determine where prices are high and low relative to each other.
    - These bands are composed of three lines: a simple moving average (the middle band) and an upper and lower band.
    - The upper and lower bands are typically two standard deviations above or below a 20-period simple moving average (SMA).
    - The bands widen and narrow as the volatility of the underlying asset changes.
    '''
    df['sma'] = df['Close'].rolling(window=window).mean()
    df['std'] = df['Close'].rolling(window=window).std()
    df[f'bollinger_upper_{window}'] = df['sma'] + (df['std'] * std)
    df[f'bollinger_lower_{window}'] = df['sma'] - (df['std'] * std)
    return df

# Calculate keltner channels
def get_keltner_channels(df, window=21, atr_window=10, atr_multiplier=2):
    '''
    - Keltner Channels are volatility-based bands that are placed on either side of an asset's price and can aid 
        in determining the direction of a trend.
    - The exponential moving average (EMA) of a Keltner Channel is typically 20 periods, although this can be adjusted if desired.
    - The upper and lower bands are typically set two times the average true range (ATR) above and below the EMA, 
        although the multiplier can also be adjusted based on personal preference.
    - Price reaching the upper Keltner Channel band is bullish while reaching the lower band is bearish.
    - The angle of the Keltner Channel also aids in identifying the trend direction. 
        The price may also oscillate between the upper and lower Keltner Channel bands, 
        which can be interpreted as resistance and support levels.
    '''
    df['ema'] = df['Close'].ewm(span=window).mean()
    df['atr'] = df['Close'].diff().abs().ewm(span=atr_window).mean()
    df[f'keltner_upper_{window}'] = df['ema'] + df['atr'] * atr_multiplier
    df[f'keltner_lower_{window}'] = df['ema'] - df['atr'] * atr_multiplier
    return df

# Calculate Donchian Channels
def get_donchian_channels(df, window=21):
    '''
    - Donchian Channels are a technical indicator that seeks to identify bullish and bearish extremes that favor reversals, 
        higher and lower breakouts, breakdowns, and other emerging trends.
    - The middle band computes the average between the highest high over a given period and the lowest low over the same time. 
        These points identify the median or mean reversion price.
    - Combining moving averages, volume indicators, and moving average convergence divergence (MACD) with Donchian channels 
        can lead to a more complete picture of the market for an asset.
    - The channels are popular for their simplicity and effectiveness, particularly for following trends and using momentum 
        strategies. They can be applied to many markets, including stocks, commodities, and forex.
    '''
    df[f'donchian_upper_{window}'] = df['High'].rolling(window=window).max()
    df[f'donchian_lower_{window}'] = df['Low'].rolling(window=window).min()
    df[f'donchian_middle_{window}'] = (df[f'donchian_upper_{window}'] + df[f'donchian_lower_{window}']) / 2
    return df

# Calculate 800-day MA
def get_800_day_ma(df):
    df['800_day_ma'] = df['Close'].rolling(window=800).mean()
    return df


# Plot daily changes of a stock: green if positive, grey if negative
# NOTE: deprecated as of 11/11/24
def plot_daily_changes(stock, data, today_data, daily_table):

    # Get start and end times from SPY
    start_time = pd.to_datetime(today_data['SPY'].index[0])
    end_time = pd.to_datetime(today_data['SPY'].index[-1])
    sub_df = today_data[stock].copy()
    sub_df = sub_df.loc[start_time:end_time]

    # Get yesterday's close
    today = pd.to_datetime(end_time.date())
    days = data[stock].index
    yesterday = days[days < today][-1]
    yesterday_close = data[stock].loc[yesterday, 'Close']

    # Plot candlestick closes for today
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot(sub_df.index, sub_df['Close'], color='black', alpha=0.8, linewidth=1.0, marker='o', markersize=4)

    # Horizontal line on yesterday's close
    ax.axhline(yesterday_close, color='black', linestyle='--', alpha=0.5, label='Yesterday Close')

    # Fill between yesterday's close and today's data
    ax.fill_between(sub_df.index, yesterday_close, sub_df['Close'], where=sub_df['Close'] > yesterday_close, color='lime', alpha=0.3)
    ax.fill_between(sub_df.index, yesterday_close, sub_df['Close'], where=sub_df['Close'] < yesterday_close, color='grey', alpha=0.2)

    # Only make x ticks on hour
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    prefix = '' if daily_table.loc[stock, 'Daily Change'] < 0 else '+'
    ax.set_title(f'{stock} Daily Change ({prefix}{daily_table.loc[stock, "Daily % Change"]:.2f}%)', fontsize=20)

    ax.legend(fontsize=20)
    ax.grid(alpha=0.3)
    # Set x tick label font size
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)

    # Save
    filepath = os.path.join('fportfolio', 'images', f'{stock}_daily.png')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


# Plot portfolio overview
def plot_portfolio_overview(data, daily_table, today_data):

    plot_axs_str = [
        ['00', '00', '00', '01']
    ]

    # Add axs_str rows for each stock trends
    num_cols = 2
    num_addl_rows = np.ceil(len(daily_table) / num_cols) 
    start_row = 2
    for i in range(int(num_addl_rows)):
        row = start_row + i
        axs_str_row = [f'{row}0', f'{row}0', f'{row}1', f'{row}1']
        plot_axs_str.append(axs_str_row)


    # Graph name to axs name
    graph_to_axs_name = {
        'daily_dollar_change': '00',
        '3_mo_pc_change_port': '01'
    }

    # Add each stock to graph_to_axs_name
    for i, stock in enumerate(daily_table.index):
        row = start_row + i // num_cols
        graph_to_axs_name[stock] = f'{row}{i % num_cols}'

    # Create subplots
    fig, axs = plt.subplot_mosaic(plot_axs_str, figsize=(30, 12 + 8 * num_addl_rows))

    graph_to_axs = {
        name: axs[graph_to_axs_name[name]] 
        for name in graph_to_axs_name
    }
    graph_to_axs['VIX'] = graph_to_axs['3_mo_pc_change_port'].twinx()

    # Plot pnl change over today
    # bar graph for each stock with daily change
    stocks, daily_changes = [], []
    daily_pc_changes = []
    unrealized_returns = []
    unrealized_pc_returns = []
    for stock in daily_table.index:
        stocks.append(stock)
        daily_changes.append(daily_table.loc[stock, 'Daily Change'])
        daily_pc_changes.append(daily_table.loc[stock, 'Daily % Change'])
        unrealized_returns.append(daily_table.loc[stock, 'Unrealized Return'])
        unrealized_pc_returns.append(daily_table.loc[stock, 'Unrealized % Return'])

    # Calculate portfolio value throughout the day
    start_time = pd.to_datetime(today_data['SPY'].index[0])
    end_time = pd.to_datetime(today_data['SPY'].index[-1])

    # Get all inds
    all_inds = set()
    for stock in daily_table.index:
        if stock == 'Portfolio': continue
        sub_df = today_data[stock].copy()
        sub_df.index = pd.to_datetime(sub_df.index)
        sub_df = sub_df.loc[start_time:end_time]
        all_inds = all_inds.union(set(sub_df.index))
    all_inds = sorted(list(all_inds))

    portfolio_df = pd.DataFrame()
    for stock in daily_table.index:
        if stock == 'Portfolio': continue
        q = daily_table.loc[stock, 'Qty']
        sub_df = today_data[stock].copy()
        sub_df.index = pd.to_datetime(sub_df.index)
        sub_df = sub_df.loc[start_time:end_time]

        # BTC very frequently has missing data for 1min so get
        # indices from SPY and then fill Nan with prev values
        sub_df = sub_df.reindex(all_inds)
        sub_df = sub_df.fillna(method='bfill')

        lows = sub_df.loc[start_time:end_time, 'Low'] * q
        highs = sub_df.loc[start_time:end_time, 'High'] * q
        opens = sub_df.loc[start_time:end_time, 'Open'] * q
        closes = sub_df.loc[start_time:end_time, 'Close'] * q
        
        if len(portfolio_df) == 0:
            portfolio_df = pd.DataFrame({'High': highs, 'Low': lows, 'Open': opens, 'Close': closes})
        else:
            portfolio_df['High'] += highs
            portfolio_df['Low'] += lows
            portfolio_df['Open'] += opens
            portfolio_df['Close'] += closes

    last_time_on_graph = portfolio_df.index[-1].strftime('%I:%M %p')
    last_date_on_graph = portfolio_df.index[-1].strftime('%m-%d')
    day_of_week = portfolio_df.index[-1].strftime('%A')

    # Plot portfolio $$$ changes ================================================================================
    graph_to_axs['daily_dollar_change'].set_title(f'{day_of_week} {last_time_on_graph} EST ({last_date_on_graph})', fontsize=30)
    today = pd.to_datetime(end_time.date())
    days = data['SPY'].index
    yesterday = days[days < today][-1]
    yesterday_closes = {s: data[s].loc[yesterday, 'Close'] for s in daily_table.index if s != 'Portfolio'}
    yesterday_close = sum([yesterday_closes[s] * daily_table.loc[s, 'Qty'] for s in yesterday_closes])

    # Colors and line width
    palette = ['tab:blue', 'tab:orange', 'tab:green', 'darkviolet', 'lime', 'tab:pink', 'gold']
    max_lw = 2.5
    min_lw = 1.3

    # Plot Daily Change $ ================================================================================
    # Calculate gain for today
    portfolio_gains = portfolio_df['Close'] - yesterday_close
    max_gain = portfolio_gains.max(); max_gain_time = portfolio_gains.idxmax()
    max_loss = portfolio_gains.min(); max_loss_time = portfolio_gains.idxmin()
    pnl_range = max_gain - max_loss
    portfolio_change_today = daily_table.loc['Portfolio', 'Daily Change']
    graph_to_axs['daily_dollar_change'].plot(portfolio_df.index, portfolio_gains, color='black', label='Portfolio', linewidth=max_lw)
    # Annotate with max gain, max loss, final change
    formatted_max = f'+{max_gain:,.0f}' if max_gain >= 0 else f'({-max_gain:,.0f})'
    graph_to_axs['daily_dollar_change'].text(max_gain_time, max_gain+pnl_range/30, formatted_max, ha='left', va='center', fontsize=18)
    formatted_min = f'+{max_loss:,.0f}' if max_loss >= 0 else f'({-max_loss:,.0f})'
    graph_to_axs['daily_dollar_change'].text(max_loss_time, max_loss-pnl_range/30, formatted_min, ha='left', va='center', fontsize=18)
    # Total $ change today
    formatted_x = f'+{portfolio_change_today:,.0f}' if portfolio_change_today >= 0 else f'({-portfolio_change_today:,.0f})'
    total_minutes = (portfolio_df.index[-1] - portfolio_df.index[0]).seconds / 60 # Timestamp
    graph_to_axs['daily_dollar_change'].text(portfolio_df.index[-1]+0.004*pd.Timedelta(minutes=total_minutes), portfolio_change_today, formatted_x, ha='left', va='center', fontsize=21)
    # Plot individual stocks
    count = 0
    for stock in daily_table.index:
        if stock == 'Portfolio': continue
        # Calculate yesterday's close
        q = daily_table.loc[stock, 'Qty']
        sub_df = today_data[stock].copy()
        sub_df.index = pd.to_datetime(sub_df.index)
        sub_df = sub_df.loc[start_time:end_time]
        gains = sub_df['Close'] * q - yesterday_closes[stock] * q
        # Calculate lw, the value / portfolio value
        ratio = daily_table.loc[stock, 'Total Value'] / daily_table.loc['Portfolio', 'Total Value']
        lw = min_lw + (max_lw - min_lw) * ratio ** 0.5
        graph_to_axs['daily_dollar_change'].plot(sub_df.index, gains, color=palette[count], label=stock, linewidth=lw)
        count += 1

    graph_to_axs['daily_dollar_change'].grid(alpha=0.8)
    graph_to_axs['daily_dollar_change'].set_ylabel('Daily Changes ($)', fontsize=24)
    graph_to_axs['daily_dollar_change'].legend(fontsize=18)
    # set x tick label font size
    graph_to_axs['daily_dollar_change'].xaxis.set_tick_params(labelsize=18)
    graph_to_axs['daily_dollar_change'].yaxis.set_tick_params(labelsize=14)
    # Format x axis labels to be hours
    graph_to_axs['daily_dollar_change'].xaxis.set_major_locator(mdates.HourLocator())
    graph_to_axs['daily_dollar_change'].xaxis.set_major_formatter(mdates.DateFormatter('%I:%M %p'))

    # 3-month moves ================================================================================
    # Plot last 3-months as a percent change from start
    three_month_back = today - datetime.timedelta(days=365*3//12)
    # Plot, calculate portfolio
    portfolio_value_3mo = pd.DataFrame()
    for stock in daily_table.index:
        if stock == 'Portfolio': continue
        q = daily_table.loc[stock, 'Qty']
        sub_df = data[stock].copy()
        sub_df = sub_df.loc[three_month_back:]
        gains = sub_df['Close'] * q
        if len(portfolio_value_3mo) == 0:
            portfolio_value_3mo = pd.DataFrame({'Close': gains})
        else:
            portfolio_value_3mo['Close'] += gains
    start = portfolio_value_3mo['Close'].iloc[0]
    changes = 100 * (portfolio_value_3mo['Close']-start) / start

    # Plot the VIX before 3mo portfolio changes so it's background
    vix_3mo = data['VIX'].loc[three_month_back:]
    graph_to_axs['VIX'].bar(vix_3mo.index, vix_3mo.loc[:, 'Close'], color='navy', alpha=0.2, label='20 < VIX < 30')
    graph_to_axs['VIX'].legend(fontsize=18, loc='upper right')
    graph_to_axs['VIX'].yaxis.set_tick_params(labelsize=18)

    # Plot portfolio on TOP ROW by itself
    graph_to_axs['3_mo_pc_change_port'].plot(portfolio_value_3mo.index, changes, label=f'Portfolio', color='black', linewidth=2.8, marker='o', markersize=4, alpha=0.8)
    graph_to_axs['3_mo_pc_change_port'].set_title(f'3-Month % Change', fontsize=24)
    graph_to_axs['3_mo_pc_change_port'].grid(alpha=0.7)
    graph_to_axs['3_mo_pc_change_port'].legend(fontsize=18, loc='upper left')
    graph_to_axs['3_mo_pc_change_port'].xaxis.set_tick_params(labelsize=18)

    # Format x axis labels to be months as names not numbers
    graph_to_axs['3_mo_pc_change_port'].xaxis.set_major_locator(mdates.MonthLocator())
    graph_to_axs['3_mo_pc_change_port'].xaxis.set_major_formatter(mdates.DateFormatter('%B'))
    # Set y tick label font size
    graph_to_axs['3_mo_pc_change_port'].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    graph_to_axs['3_mo_pc_change_port'].yaxis.set_tick_params(labelsize=18)


    # Plot each stock's trends ================================================================================
    funcs = [get_bollinger_bands, get_keltner_channels, get_donchian_channels]
    one_year_back = today - datetime.timedelta(days=365)
    window = 50
    for stock in daily_table.index:
        # Calculate indicators (50 day)
        df = data[stock].copy()
        for func in funcs:
            df = func(df, window=window)
        df = df.loc[one_year_back:]

        # Plot purchased price
        graph_to_axs[stock].axhline(daily_table.loc[stock, 'Purchase Price'], color='black', linestyle=':', label=None, alpha=1.0, linewidth=3.0)

        # Plot price for each day for the past year
        graph_to_axs[stock].plot(df.index, df['Close'], color='black', label=f'{stock} 1-Year Prices', alpha=0.8, linewidth=2.0, marker='o', markersize=3)

        # Plot indicators for 1-year
        alpha = 0.6
        linewidth = 1.0
        # Plot bollinger bands
        graph_to_axs[stock].plot(df.index, df[f'bollinger_upper_{window}'], color='blue', label=f'Bollinger {window} (vol)', alpha=alpha, linewidth=linewidth)
        graph_to_axs[stock].plot(df.index, df[f'bollinger_lower_{window}'], color='blue', alpha=alpha, linewidth=linewidth)

        # Plot keltner channels
        graph_to_axs[stock].plot(df.index, df[f'keltner_upper_{window}'], color='orange', label=f'Keltner {window} (trend)', alpha=0.8, linewidth=linewidth)
        graph_to_axs[stock].plot(df.index, df[f'keltner_lower_{window}'], color='orange', alpha=0.8, linewidth=linewidth)

        # Plot donchian channels
        graph_to_axs[stock].plot(df.index, df[f'donchian_upper_{window}'], color='deeppink', label=f'Donchian {window} (hi/lo)', alpha=alpha, linewidth=linewidth)
        graph_to_axs[stock].plot(df.index, df[f'donchian_lower_{window}'], color='deeppink', alpha=alpha, linewidth=linewidth)

        # Formatting
        graph_to_axs[stock].legend(fontsize=18)
        graph_to_axs[stock].grid(alpha=0.3)
        # Format x axis labels to be months
        graph_to_axs[stock].xaxis.set_major_locator(mdates.MonthLocator())
        graph_to_axs[stock].xaxis.set_major_formatter(mdates.DateFormatter('%b')) # abbreviated month
        graph_to_axs[stock].set_ylabel(stock, fontsize=24)
        # X, Y tick label font size
        graph_to_axs[stock].xaxis.set_tick_params(labelsize=18)
        graph_to_axs[stock].yaxis.set_tick_params(labelsize=14)


    # For each ax, extend y-lim so that annotations don't get cut off
    addition = 0.05
    for key in axs:
    # for ax in axs:
        ax = axs[key]
        delta = (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.set_ylim(ax.get_ylim()[0] - addition * delta, ax.get_ylim()[1] + addition * delta)
        # Add horizontal line at 0
        ax.axhline(0, color='black', linestyle='-', alpha=0.5)    
    
    plt.tight_layout()
    filepath = os.path.join('fportfolio', 'images', 'portfolio_overview.png')
    plt.savefig(filepath)
    plt.close()


# 3 plots for each stock: 3-month, 1-year, 5-year
def plot_trends(df, name, positions):

    # We only want within the past x days
    today = datetime.datetime.today().date()
    # five years back
    five_years_back = today - datetime.timedelta(days=5*365)
    # one year back
    one_year_back = today - datetime.timedelta(days=365)
    # 3 month back
    three_month_back = today - datetime.timedelta(days=365*3//12)
    # Since we only looking back 5 year max using 50 day window
    long_time_back = today - datetime.timedelta(days=(5*365) + 50)

    df = df.loc[long_time_back:].copy()

    # Calculate indicators (daily)
    df = get_bollinger_bands(df, window=21)
    df = get_bollinger_bands(df, window=50)
    df = get_keltner_channels(df, window=21)
    df = get_keltner_channels(df, window=50)
    df = get_donchian_channels(df, window=21)
    df = get_donchian_channels(df, window=50)
    df = get_800_day_ma(df)

    three_month = df.loc[three_month_back:]
    one_year = df.loc[one_year_back:]
    five_year = df.loc[five_years_back:]
    dfs = [three_month, one_year, five_year]

    # Plot each time period
    fig, axs = plt.subplots(3, 1, figsize=(20, 30), sharex=False)
    for i, df_x in enumerate(dfs):
        # 3 month
        if i == 0:
            window = 21
            pc_change = df_x.iloc[-1]['Close'] / df_x.iloc[0]['Close'] - 1
            prefix = '' if pc_change < 0 else '+'
            axs[i].set_title(f'{name} 3-month ({prefix}{pc_change*100:,.2f}%)', fontsize=20)
            alpha = 0.6
            linewidth = 1.0
        # 1 year
        elif i == 1:
            window = 50
            pc_change = df_x.iloc[-1]['Close'] / df_x.iloc[0]['Close'] - 1
            prefix = '' if pc_change < 0 else '+'
            axs[i].set_title(f'{name} 1-year ({prefix}{pc_change*100:,.2f}%)', fontsize=20)
            alpha = 0.6
            linewidth = 1.0
        # 5 year
        elif i == 2:
            window = 50
            pc_change = df_x.iloc[-1]['Close'] / df_x.iloc[0]['Close'] - 1
            prefix = '' if pc_change < 0 else '+'
            axs[i].set_title(f'{name} 5-year ({prefix}{pc_change*100:,.2f}%)', fontsize=20)
            alpha = 0.6
            linewidth = 0.7

        # Plot purchased price
        axs[i].axhline(positions[name]['purchased_price'], color='black', linestyle=':', label='Purchased Price', alpha=1.0, linewidth=3.0)

        # Plot line
        axs[i].plot(df_x.index, df_x['Close'], color='black', label='Price (Close)', alpha=0.8, linewidth=2.0, marker='o', markersize=6/(i+1))

        # Plot trend lines
        if window is not None:
            # Plot bollinger bands
            axs[i].plot(df_x.index, df_x[f'bollinger_upper_{window}'], color='blue', label=f'Bollinger {window} (vol)', alpha=alpha, linewidth=linewidth)
            axs[i].plot(df_x.index, df_x[f'bollinger_lower_{window}'], color='blue', alpha=alpha, linewidth=linewidth)

            # Plot keltner channels
            axs[i].plot(df_x.index, df_x[f'keltner_upper_{window}'], color='orange', label=f'Keltner {window} (trend)', alpha=0.8, linewidth=linewidth)
            axs[i].plot(df_x.index, df_x[f'keltner_lower_{window}'], color='orange', alpha=0.8, linewidth=linewidth)

            # Plot donchian channels
            axs[i].plot(df_x.index, df_x[f'donchian_upper_{window}'], color='deeppink', label=f'Donchian {window} (hi/lo)', alpha=alpha, linewidth=linewidth)
            axs[i].plot(df_x.index, df_x[f'donchian_lower_{window}'], color='deeppink', alpha=alpha, linewidth=linewidth)


        axs[i].legend(fontsize=18)
        axs[i].grid(alpha=0.3)

        if i == 0:
            # Format x axis labels to be months
            axs[i].xaxis.set_major_locator(mdates.MonthLocator())
            axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%B'))
        elif i == 1:
            # Format x axis labels to be months
            axs[i].xaxis.set_major_locator(mdates.MonthLocator())
            axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%b')) # abbreviated month
        elif i == 2:
            pass

    # Change x-label, y-label fontsize
    for ax in axs:
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)


    plt.tight_layout()
    filename = os.path.join('fportfolio', 'images', f'{name}.png')
    plt.savefig(filename)
    plt.close()



# Calculate pearson correlation
# and plot heatmap
def pearson_corr(daily_table, data):
    # Calculate pearson correlation for all combos of stock
    # and plot correlation matrix
    corrs = np.zeros((len(daily_table)-1, len(daily_table)-1))
    c1 = 0
    for stock in daily_table.index:
        if stock == 'Portfolio': continue
        stock_data = data[stock]['Close'].pct_change()
        c2 = 0
        for stock2 in daily_table.index:
            if stock2 == 'Portfolio': continue
            if stock == stock2: corr = 1
            else:
                stock2_data = data[stock2]['Close'].pct_change()
                corr = stock_data.corr(stock2_data)
            corrs[c1][c2] = corr
            c2 += 1
        c1 += 1

    # Add to daily_table: each stock's corr w/ SPY
    for stock in daily_table.index:
        if stock == 'Portfolio': continue
        stock_data = data[stock]['Close'].pct_change()
        spy_data = data['SPY']['Close'].pct_change()
        corr = stock_data.corr(spy_data)
        daily_table.loc[stock, 'SPY Corr'] = corr
    # Calculate portfolio corr w/ SPY
    # Weighted average of each stock's corr w/ SPY
    daily_table.loc['Portfolio', 'SPY Corr'] = 0
    for stock in daily_table.index:
        if stock == 'Portfolio': continue
        frac_portfolio = daily_table.loc[stock, '% Portfolio']/100
        corr = daily_table.loc[stock, 'SPY Corr']
        daily_table.loc['Portfolio', 'SPY Corr'] += frac_portfolio * corr

    # Get order of stocks based on total value
    order = daily_table.sort_values(by='Total Value', ascending=False).index
    order = [x for x in order if x != 'Portfolio']

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(corrs, cmap='viridis')
    # Label with stock names
    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_yticklabels(order)
    # Add colorbar with values
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Correlation', rotation=-90, va='bottom')
    # Add annotation of values
    for i in range(len(order)):
        for j in range(len(order)):
            value = corrs[i, j]
            if value > 0.5: color = 'black'
            else: color = 'white'
            text = ax.text(j, i, f'{value:.2f}', ha='center', va='center', color=color)
    # Save
    filepath = os.path.join('fportfolio', 'images', 'corr_matrix.png')
    plt.savefig(filepath)
    plt.close()

    return corrs
