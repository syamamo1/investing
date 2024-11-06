import yfinance as yf
import pickle
import argparse
import pandas as pd
from tqdm import tqdm
import datetime
import copy
import os
import time
import numpy as np
import urllib.request

from plotters import plot_trends, plot_portfolio_overview, pearson_corr
from html_generator import generate_html_page, make_nice_string
from positions import get_positions

# Save data to pickle
def save_data(stuff, filepaths):
    for name in stuff:
        filepath = filepaths[name]
        data = stuff[name]
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)


# Load data from pickle
def load_data(filepaths):
    stuff = {}
    for name in filepaths:
        filepath = filepaths[name]
        with open(filepath, 'rb') as f:
            stuff[name] = pickle.load(f)
    return stuff


# Check internet connection
def has_internet_connection():
    try:
        # Try to connect to a website (Google in this case)
        urllib.request.urlopen('https://www.google.com/', timeout=5)
        print('Internet connection detected')
        return True
    except urllib.error.URLError as err:
        print('No internet connection detected')
        return False


# Get fundamental data
def get_fundamental_data(stocks):
    fundamental_data = {}
    for stock in stocks:
        try:
            ticker = yf.Ticker(stock)
            info = ticker.info
            fundamental_data[stock] = info

        except Exception as e:
            print(f'Failed to get fundamental data for {stock}')
            continue
    return fundamental_data


# Download from online
def download_daily_data(stocks, start_date='2016-01-01'):
    stocks = list(stocks)
    today = datetime.datetime.today()
    today = today.strftime('%Y-%m-%d')

    # One call to download all data
    # Price        Adj Close                                         ...      Volume                               
    # Ticker            AAPL       BTC-USD        MSFT         NVDA  ...        NVDA        QCOM          SPY SWPPX
    # Date                                                           ...                                           
    # 2016-01-01         NaN    434.334015         NaN          NaN  ...         NaN         NaN          NaN   NaN
    # 2016-01-02         NaN    433.437988         NaN          NaN  ...         NaN         NaN          NaN   NaN
    # 2016-01-03         NaN    430.010986         NaN          NaN  ...         NaN         NaN          NaN   NaN
    # 2016-01-04   23.914476    433.091003   48.521477     7.899905  ...  35807600.0  12571300.0  222353500.0   0.0
    # 2016-01-05   23.315199    431.959991   48.742832     8.026808  ...  49027200.0  13482300.0  110845800.0   0.0
    # ...                ...           ...         ...          ...  ...         ...         ...          ...   ...
    # 2024-05-23  186.880005  67929.562500  427.000000  1037.989990  ...  83506500.0  14475100.0   57211200.0   0.0
    # 2024-05-24  189.979996  68526.101562  430.160004  1064.689941  ...  42650200.0  13799000.0   41258400.0   0.0
    # 2024-05-25         NaN  69265.945312         NaN          NaN  ...         NaN         NaN          NaN   NaN
    # 2024-05-26         NaN  68518.093750         NaN          NaN  ...         NaN         NaN          NaN   NaN
    # 2024-05-27         NaN  69394.554688     
    #     NaN          NaN  ...         NaN         NaN          NaN   NaN
    raw_data = yf.download(stocks, start_date, today)
    data = {}
    for stock in stocks:
        # Select stock from multi-index: any index with index = (x, stock)
        stock_data = raw_data.loc[:, (slice(None), stock)]
        # Remove multi-index
        stock_data.columns = stock_data.columns.droplevel(1)
        # Remove rows with NaN
        stock_data = stock_data.dropna()
        # Index to datetime
        stock_data.index = pd.to_datetime(stock_data.index)
        data[stock] = stock_data
        fn = os.path.join('fportfolio', 'data', f'{stock}.pkl')
        with open(fn, 'wb') as f:
            pickle.dump(stock_data, f)

    return data

# Get this month's data with 1hr interval
def get_month_data(stocks):
    stocks = list(stocks)

    tickers = yf.Tickers(' '.join(stocks))
    data = {}
    for stock in stocks:
        # Select stock from multi-index: any index with index = (x, stock)
        #                                 Open       High        Low      Close  Volume  Dividends  Stock Splits  Capital Gains
        # Datetime                                                                                                             
        # 2024-05-01 09:30:00-04:00  77.239998  77.239998  77.239998  77.239998       0        0.0           0.0            0.0
        # 2024-05-02 09:30:00-04:00  77.940002  77.940002  77.940002  77.940002       0        0.0           0.0            0.0
        # 2024-05-03 09:30:00-04:00  78.930000  78.930000  78.930000  78.930000       0        0.0           0.0            0.0
        # 2024-05-06 09:30:00-04:00  79.739998  79.739998  79.739998  79.739998       0        0.0           0.0            0.0
        # 2024-05-07 09:30:00-04:00  79.849998  79.849998  79.849998  79.849998       0        0.0           0.0            0.0
        # 2024-05-08 09:30:00-04:00  79.849998  79.849998  79.849998  79.849998       0        0.0           0.0            0.0
        # 2024-05-09 09:30:00-04:00  80.279999  80.279999  80.279999  80.279999       0        0.0           0.0            0.0
        # 2024-05-10 09:30:00-04:00  80.419998  80.419998  80.419998  80.419998       0        0.0           0.0            0.0
        # 2024-05-13 09:30:00-04:00  80.400002  80.400002  80.400002  80.400002       0        0.0           0.0            0.0
        # 2024-05-14 09:30:00-04:00  80.800003  80.800003  80.800003  80.800003       0        0.0           0.0            0.0
        # 2024-05-15 09:30:00-04:00  81.760002  81.760002  81.760002  81.760002       0        0.0           0.0            0.0
        stock_data = tickers.tickers[stock].history(period='1mo', interval='1h')
        # Index like: 2024-05-28 09:30:00-04:00
        # Convert index to string, remove timezone
        stock_data.index = stock_data.index.strftime('%Y-%m-%d %H:%M:%S')
        # Convert to datetime
        stock_data.index = pd.to_datetime(stock_data.index)
        data[stock] = stock_data

    return data


# Get today's data with 1min interval
def get_today_data(stocks):
    stocks = list(stocks)
    
    data = {}
    for stock in stocks:
        # SWPPX special case, not available on day of
        if stock == 'SWPPX': continue
        # Get today's data
        ticker = yf.Ticker(stock)
        if stock == 'BTC-USD': today_data = ticker.history(period='5d', interval='1m')
        else: today_data = ticker.history(period='1d', interval='1m')
        # Index like: 2024-05-28 09:30:00-04:00
        # Convert index to string, remove timezone
        today_data.index = today_data.index.strftime('%Y-%m-%d %H:%M:%S')
        # Convert to datetime
        today_data.index = pd.to_datetime(today_data.index)
        data[stock] = today_data

    return data

# Load from local pickle
def load_daily_data(stocks, start_date='2016-01-01'):
    data = {}
    for stock in stocks:
        fn = os.path.join('fportfolio', 'data', f'{stock}.pkl')
        with open(fn, 'rb') as f:
            data[stock] = pickle.load(f)
            # Filter out data before start_date
            data[stock] = data[stock].loc[start_date:]
        print(f'Loaded {fn}')
    return data


# Gets most recent quote for stock
def get_current_prices(stocks):
    current_prices = {}
    for stock in stocks:
        ticker = yf.Ticker(stock)
        if stock != 'SWPPX': 
            last_quote = ticker.history(interval='1m', period = '1d')
            last_quote = last_quote.iloc[-1]['Close']
        # SWPPX special case, on day of, can only load 
        # data up to yesterday. So get yesterday's quote
        elif stock == 'SWPPX':
            last_quote = ticker.history().iloc[-1]['Close']

        current_prices[stock] = last_quote
        
    return current_prices

# Gets yesterday's quote for stock
def get_yesterday_prices(stocks):
    yesterday_prices = {}
    for stock in stocks:
        ticker = yf.Ticker(stock)
        last_quote = ticker.history(interval='1d', period = '1mo')
        last_quote = last_quote.iloc[-2]['Close']

        yesterday_prices[stock] = last_quote
        
    return yesterday_prices

# Gets last week's price for stock
def get_last_week_prices(stocks):
    last_week_prices = {}
    for stock in stocks:
        ticker = yf.Ticker(stock)
        last_quote = ticker.history(interval='1d', period = '1mo')
        # Last close 5 days back
        last_quote = last_quote.iloc[-6]['Close']

        last_week_prices[stock] = last_quote
        
    return last_week_prices


# Combine SPY IRA + SPY edge
def combine_positions(positions):
    temp_positions = copy.deepcopy(positions)
    positions = {}

    # Combine SPY, NDAQ
    counts = {}
    for stock in temp_positions:
        name = stock.split()[0]
        if name in counts:
            counts[name] += 1
        else:
            counts[name] = 1
    
    done = []
    for stock in temp_positions:
        if stock in done: continue
        name = stock.split()[0]
        if counts[name] == 1:
            positions[stock] = temp_positions[stock]
        else:
            edge = f'{name} edge'
            IRA = f'{name} IRA'
            if edge in temp_positions and IRA in temp_positions:
                positions[name] = {
                    'purchased_price': (temp_positions[edge]['purchased_price'] * temp_positions[edge]['quantity'] + 
                                        temp_positions[IRA]['purchased_price'] * temp_positions[IRA]['quantity']) / 
                                        (temp_positions[edge]['quantity'] + temp_positions[IRA]['quantity']),
                    'quantity': temp_positions[edge]['quantity'] + temp_positions[IRA]['quantity']
                }
                done.append(edge)
                done.append(IRA)
            elif edge in temp_positions:
                positions[edge] = temp_positions[edge]
            elif IRA in temp_positions:
                positions[IRA] = temp_positions[IRA]

    return positions


# Add ubiquity SWPPX to SPY position
def add_ubiquity(positions, ubiquity, data):
    # Calculate purchased_price, quantity
    for stock in ubiquity.keys():
        value = ubiquity[stock]['value']
        unrealized_gain = ubiquity[stock]['unrealized_gain']
        quantity = ubiquity[stock]['quantity']
        ubiquity[stock] = {
            'purchased_price': (value-unrealized_gain) / quantity,
            'quantity': quantity
        }
    
    # Calculate multiplier
    # Get SWPPX prices
    spy = data['SPY']['Close']
    swppx = data['SWPPX']['Close']

    # Get multiplier
    multipliers = []
    for date in set(spy.index).intersection(set(swppx.index)):
        spy_price = spy.loc[date]
        swppx_price = swppx.loc[date]
        multiplier = spy_price / swppx_price # how many swppx per spy (> 1)
        multipliers.append(multiplier)
    multiplier = sum(multipliers) / len(multipliers)

    # random_date = set(spy.index).intersection(set(swppx.index)).pop()
    # spy_price = spy.loc[random_date]
    # swppx_price = swppx.loc[random_date]
    # multiplier = spy_price / swppx_price # how many swppx per spy (> 1)

    # Convert to SPY
    ub_pp = ubiquity['SWPPX']['purchased_price'] * multiplier
    ub_q = ubiquity['SWPPX']['quantity'] / multiplier

    # Add to positions
    cur_pp = positions['SPY']['purchased_price']
    cur_q = positions['SPY']['quantity']
    total_value = cur_pp * cur_q + ub_pp * ub_q
    total_q = cur_q + ub_q
    new_pp = total_value / total_q

    positions['SPY'] = {
        'purchased_price': new_pp,
        'quantity': total_q
    }
    
    return positions


# Daily analysis: show daily gain, total gain
def calculate_daily_table(positions, data, fundamental_data):
    daily_table = copy.deepcopy(positions)
    current_prices = get_current_prices(positions.keys())
    yesterday_prices = get_yesterday_prices(positions.keys())

    # Calculate total
    total_current_price = 0
    total_yesterday_price = 0
    total_purchased_price = 0
    total_value = 0
    for stock in positions:
        current_price = current_prices[stock]
        yesterday_price = yesterday_prices[stock]
        purchased_price = positions[stock]['purchased_price']
        quantity = positions[stock]['quantity']

        # Add to total
        total_current_price += current_price * quantity
        total_yesterday_price += yesterday_price * quantity
        total_purchased_price += purchased_price * quantity
        total_value += current_price * quantity

    # Calculate total gains as percentage %
    total_daily_gain = (total_current_price - total_yesterday_price)
    total_total_gain = (total_current_price - total_purchased_price)
    total_daily_pc_gain = 100 * (total_current_price - total_yesterday_price) / total_yesterday_price
    total_total_pc_gain = 100 * (total_current_price - total_purchased_price) / total_purchased_price

    # Add to table
    daily_table['Portfolio'] = {
        'Total Value': total_value,
        '% Portfolio': 100 * total_value / total_value,
        'Qty': 1,
        'Cur. Price': total_current_price,
        'Purchase Price': total_purchased_price,
        'Daily Change': total_daily_gain,
        'Daily % Change': total_daily_pc_gain,
        'Unrealized Return': total_total_gain,
        'Unrealized % Return': total_total_pc_gain
    }
    
    # Calculate individual stocks
    for stock in positions:
        current_price = current_prices[stock]
        yesterday_price = yesterday_prices[stock]
        purchased_price = positions[stock]['purchased_price']
        quantity = positions[stock]['quantity']

        # Calculate gains
        daily_gain = (current_price - yesterday_price) * quantity
        total_gain = (current_price - purchased_price) * quantity
        daily_pc_gain = 100 * (current_price - yesterday_price) / yesterday_price
        total_pc_gain = 100 * (current_price - purchased_price) / purchased_price
        daily_table[stock]['Total Value'] = current_price * quantity
        daily_table[stock]['% Portfolio'] = 100 * (current_price * quantity) / total_value
        daily_table[stock]['Qty'] = quantity
        daily_table[stock]['Cur. Price'] = current_price
        daily_table[stock]['Purchase Price'] = purchased_price
        daily_table[stock]['Daily Change'] = daily_gain
        daily_table[stock]['Daily % Change'] = daily_pc_gain
        daily_table[stock]['Unrealized Return'] = total_gain
        daily_table[stock]['Unrealized % Return'] = total_pc_gain

        # Calculate weekly (5 days)/monthly (21 days) volatility of returns
        daily_returns = data[stock]['Close'].pct_change()
        weekly_vol = daily_returns.iloc[-5:].std() * 100 # get as pct
        monthly_vol = daily_returns.iloc[-21:].std() * 100 # get as pct
        daily_table[stock]['Weekly Vol'] = weekly_vol
        daily_table[stock]['Monthly Vol'] = monthly_vol

        # Add fundamental data
        if stock in fundamental_data:
            # These don't have PE ratios
            if stock in ['SPY', 'QQQ']:
                daily_table[stock]['Fwd PE'] = '-'
                daily_table[stock]['Trail PE'] = '-'
                daily_table[stock]['Fwd EPS'] = '-'
                daily_table[stock]['Trail EPS'] = '-'

                # trailingAnnualDividendRate = actual $ paid over last year
                annualDivRate = fundamental_data[stock]['trailingAnnualDividendRate']
                yearDiv = annualDivRate * quantity
                daily_table[stock]['Year $ Div'] = yearDiv

            # These don't have anything
            elif stock in ['BTC-USD']:
                daily_table[stock]['Fwd PE'] = '-'
                daily_table[stock]['Trail PE'] = '-'
                daily_table[stock]['Fwd EPS'] = '-'
                daily_table[stock]['Trail EPS'] = '-'
                daily_table[stock]['Year $ Div'] = 0

            # These have everything
            else:
                daily_table[stock]['Fwd PE'] = fundamental_data[stock]['forwardPE']
                daily_table[stock]['Trail PE'] = fundamental_data[stock]['trailingPE']
                daily_table[stock]['Fwd EPS'] = fundamental_data[stock]['forwardEps']
                daily_table[stock]['Trail EPS'] = fundamental_data[stock]['trailingEps']

                # Div Yield = ($ div per share) / (current price)
                frequency = 1 / fundamental_data[stock]['dividendRate']
                yearDiv = frequency * quantity * fundamental_data[stock]['dividendYield'] * daily_table[stock]['Cur. Price']
                daily_table[stock]['Year $ Div'] = yearDiv


    # Add portfolio to daily table
    daily_table['Portfolio']['Fwd PE'] = '-'
    daily_table['Portfolio']['Trail PE'] = '-'
    daily_table['Portfolio']['Fwd EPS'] = '-'
    daily_table['Portfolio']['Trail EPS'] = '-'
    daily_table['Portfolio']['Year $ Div'] = sum([daily_table[stock]['Year $ Div'] for stock in daily_table if stock != 'Portfolio'])


    # Calculate total portfolio volatility
    daily_returns = pd.Series(dtype='float64')
    for stock in positions:
        # Weight by % portfolio
        daily_returns = daily_returns.add(data[stock]['Close'].pct_change() * daily_table[stock]['% Portfolio']/100, fill_value=0)
    total_weekly_vol = daily_returns.iloc[-5:].std() * 100
    total_monthly_vol = daily_returns.iloc[-21:].std() * 100
    daily_table['Portfolio']['Weekly Vol'] = total_weekly_vol
    daily_table['Portfolio']['Monthly Vol'] = total_monthly_vol

    # Make df
    df = pd.DataFrame(daily_table).T
    df = df.sort_values(by='Total Value', ascending=False)

    
    # Print results
    print('='*70)
    cols_to_print = ['Daily Change', 'Daily % Change', 'Unrealized % Return', 'Total Value']
    print_df = df[cols_to_print].copy()
    for col in cols_to_print:
        print_df[col] = print_df[col].apply(lambda x: make_nice_string(x, col))
    print(print_df)
    print('='*70, '\n')
    return df


# Add today's data to data
# When we're in the middle of a trading day, use  
# the current price as "this day's price" in the 3-month graph
# for example
def add_today_data(data, today_data):
    columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    today = pd.to_datetime(datetime.datetime.today())
    for stock in today_data:
        df = data[stock]
        hi = today_data[stock]['High'].max()
        lo = today_data[stock]['Low'].min()
        open = today_data[stock].iloc[0]['Open']
        close = today_data[stock].iloc[-1]['Close']
        volume = today_data[stock]['Volume'].sum()
        # Add to data
        data[stock].loc[today, columns] = [open, hi, lo, close, volume]
    return data


# Add portfolio to data
def add_portfolio(data, positions):
    # Get all dates
    dates = set()
    for stock in data:
        dates = dates.union(set(data[stock].index))
    dates = sorted(list(dates))

    # Add portfolio to data
    portfolio = pd.DataFrame(index=dates, columns=['Open', 'High', 'Low', 'Close'])
    for col in portfolio.columns: portfolio[col] = 0
    for stock in positions:
        quantity = positions[stock]['quantity']
        for col in portfolio.columns:
            portfolio[col] += data[stock][col] * quantity
    # remove nan
    portfolio = portfolio.dropna()

    data['Portfolio'] = portfolio

    return data


# Historical betas
# beta = cov(stock, SPY) / var(SPY)  ==> using daily changes
# This is a measurement of how much a stock moves in relation to the market
def calculate_betas(daily_table, data):
    # Calculate betas
    betas = {}
    for stock in daily_table.index:
        if stock == 'Portfolio': continue
        stock_data = data[stock]['Close'].pct_change()
        spy_data = data['SPY']['Close'].pct_change()
        cov = stock_data.cov(spy_data)
        var = spy_data.var()
        beta = cov / var
        betas[stock] = beta

    # Get order of stocks based on total value
    order = daily_table.sort_values(by='Total Value', ascending=False).index
    order = [x for x in order if x != 'Portfolio']

    # Add beta to daily_table
    for stock in daily_table.index:
        # Calculate weighted average of beta
        if stock == 'Portfolio': 
            # Weight = frac portfolio * beta
            daily_table.loc[stock, 'β'] = 0
            for stock2 in order:
                frac_portfolio = daily_table.loc[stock2, '% Portfolio']/100
                beta = betas[stock2]
                daily_table.loc[stock, 'β'] += frac_portfolio * beta
        else:
            daily_table.loc[stock, 'β'] = betas[stock]

    return betas


# For calling main function from server
def main_wrapper():
    try:
        main()
    except Exception as e:
        print(e)
        print('Failed to run main()')

'''
TODO:
Calculate beta as linear regression of returns 
stock x and SPY (independent variable). alpha is the 
y-intercept. 
'''

def main(fast=0, stock_graphs=1):
    positions, ubiquity = get_positions()

    # Get stocks to get ======================================
    watchlist = [r'%5EVIX'] # Check out and buy this?
    stocks_to_get = list(positions.keys()) + list(ubiquity.keys()) + watchlist    

    # 2. Get data ======================================
    filepaths = {
        'data': os.path.join('fportfolio', 'data', 'data.pkl'),
        'today_data': os.path.join('fportfolio', 'data', 'today_data.pkl'),
        'month_data': os.path.join('fportfolio', 'data', 'month_data.pkl'),
        'fundamental_data': os.path.join('fportfolio', 'data', 'fundamental_data.pkl'),
        'positions': os.path.join('fportfolio', 'data', 'positions.pkl')
    }

    if has_internet_connection() and not fast: # Toggle quick generation
        data = download_daily_data(stocks_to_get, start_date = '2016-01-01') # hi/lo/open/close for each day
        month_data = get_month_data(stocks_to_get) # get monthly data 1hr data
        today_data = get_today_data(stocks_to_get) # get today's 5min data
        data = add_today_data(data, today_data) # add today's intraday data to data

        fundamental_data = get_fundamental_data(stocks_to_get) 

        # Add ubiquity to positions: convert SWPPX to SPY
        positions = add_ubiquity(positions, ubiquity, data)
        data = add_portfolio(data, positions)
        data['VIX'] = data.pop(r'%5EVIX') # Rename VIX

        # ADD MULTIPLIER SO PEOPLE CAN'T SEE TRUE VALUE
        MULTIPLIER = 1
        for stock in positions:
            positions[stock]['quantity'] *= MULTIPLIER # multiplier

        # Save daily table to pickle
        stuff = {
            'data': data,
            'today_data': today_data,
            'month_data': month_data,
            'fundamental_data': fundamental_data,
            'positions': positions
        }
        save_data(stuff, filepaths)            
    else:
        stuff = load_data(filepaths)
        data = stuff['data']
        today_data = stuff['today_data']
        month_data = stuff['month_data']
        fundamental_data = stuff['fundamental_data']
        positions = stuff['positions']
        print('Loaded daily table and data from pickle')


    # 3. daily table ======================================
    # Calculate daily table
    daily_table = calculate_daily_table(positions, data, fundamental_data)

    # 4. Plotting ======================================
    # Dailiy + Historical Portfolio performance
    plot_portfolio_overview(data, daily_table, today_data)

    # Create trend plots
    for stock in tqdm(positions, desc='Creating trend plots'):
        if not stock_graphs: continue
        plot_trends(data[stock], stock, positions)

    # Calculate betas: how much stock moves in relation to market
    # beta = cov(stock, SPY) / var(SPY)
    calculate_betas(daily_table, data)

    # Calculate pearson correlation and plot
    pearson_corr(daily_table, data)

    # Create HTML
    generate_html_page(daily_table)

    # '%Y-%m-%d %I:%M %p'
    print(f'Done ==> {datetime.datetime.now().strftime("%A %Y-%m-%d @ %I:%M:%S %p")}')
    print('='*70)



if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser()    
    parser.add_argument("--fast", type=int, 
                        default=0, 
                        help="Do you want it to reload data (0) or be fast (1)?"
                        )
    parser.add_argument("--stock-graphs", type=int, 
                        default=0, 
                        help="Do you want it to remake graphs (1) or skip (0)?"
                        )
    args = parser.parse_args()
    main(args.fast, args.stock_graphs)

    print(f'--- {time.time() - start_time:.2f} seconds ---')