import os
import datetime
from yattag import Doc
from tqdm import tqdm



# Function to add the appropriate suffix for the day
def add_suffix(day):
    if 11 <= day <= 13:
        return f"{day}th"
    else:
        suffixes = {1: 'st', 2: 'nd', 3: 'rd'}
        return f"{day}{suffixes.get(day % 10, 'th')}"


# Get definitions
def get_definitions():
    # Column definitions
    col_defs = {
        'Qty': 'Number of shares',
        'Purchase Price': 'Price stock was purchased at ($)',
        'Cur. Price': 'Current price of stock ($)',
        'β': 'Stock risk in relation to market',
        'SPY Corr': 'Correlation with SPY',
        'Fwd PE': '(Current price) to (Anticipated earnings) ratio',
        'Trail PE': '(Current price) to (Past earnings) ratio',
        'Fwd EPS': 'Anticipated earnings per share',
        'Trail EPS': 'Past earnings per share',
        'Year $ Div': 'Expected dividends payout for sum of shares',
        'Weekly Vol': 'σ of 5 day returns (%)',
        'Monthly Vol': 'σ of 21 day returns (%)',
        'Daily % Change': 'Daily percentage change',
        'Daily Change': 'Daily position dollar change ($)',
        'Unrealized % Return': 'Unrealized percentage return',
        'Unrealized Return': 'Unrealized dollar return ($)',
        '% Portfolio': 'Percentage of total portfolio value',
        'Total Value': 'Total value of position ($)'
    }
    # Investopedia definitions
    indicator_definitions = {
        'Bollinger Bands': 
        '''
        - Bollinger Bands is a technical analysis tool used to determine where prices are high and low relative to each other.
        - These bands are composed of three lines: a simple moving average (the middle band) and an upper and lower band.
        - The upper and lower bands are typically two standard deviations above or below a 20-period simple moving average (SMA).
        - The bands widen and narrow as the volatility of the underlying asset changes.
        ''',

        'Keltner Channels':
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
        ''',

        'Donchian Channels':
        '''
        - Donchian Channels are a technical indicator that seeks to identify bullish and bearish extremes that favor reversals, 
            higher and lower breakouts, breakdowns, and other emerging trends.
        - The middle band computes the average between the highest high over a given period and the lowest low over the same time. 
            These points identify the median or mean reversion price.
        - Combining moving averages, volume indicators, and moving average convergence divergence (MACD) with Donchian channels 
            can lead to a more complete picture of the market for an asset.
        - The channels are popular for their simplicity and effectiveness, particularly for following trends and using momentum 
            strategies. They can be applied to many markets, including stocks, commodities, and forex.
        ''',

        'VIX':
        '''
        - The VIX measures the market's expectation of 30-day volatility for the S&P 500 Index.
        - Calculation: It's calculated using the prices of S&P 500 index options, both puts and calls, across a range of strike prices.
        - Nickname: Often called the "fear index" or "fear gauge" as it reflects investor sentiment and market uncertainty.
        - Scale: Expressed as a percentage, representing the expected annualized change in the S&P 500 index over the next 30 days.
        - Interpretation: VIX below 20: Generally indicates low volatility and market stability, VIX 20-30: Moderate volatility, VIX above 30: High volatility, often indicating market stress or uncertainty.
        '''
    }

    definitions = {
        'col_defs': col_defs,
        'indicator_definitions': indicator_definitions
    }

    return definitions


# De-HTML the symbols &, <, >, >=, <=
# in javascript part of HTML
def dehtml(html_document_text):
    # Get to the part with javascript
    split1 = html_document_text.split('<script>')
    split2 = split1[1].split('</script>')
    js = split2[0]

    # Replace
    js = js.replace('&amp;', '&')
    js = js.replace('&lt;', '<')
    js = js.replace('&gt;', '>')
    js = js.replace('&gt;=', '>=')
    js = js.replace('&lt;=', '<=')

    # Put back together
    html_document_text = split1[0] + '<script>' + js + '</script>' + split2[1]
    return html_document_text


# Make nice string
def make_nice_string(value, col='Daily % Change'):
    if type(value) == str: return value

    is_neg = value < 0

    # Percentages
    if '%' in col: value = f'{value:.2f}%' # pct: x => x%
    
    # 2 decimal
    elif col in ['β', 'Weekly Vol', 'Monthly Vol', 'SPY Corr', 
                 'Cur. Price', 'Purchase Price', 
                 'Div Yield']: 
        value = f'{value:,.2f}' # beta, vol

    # 1 decimal
    elif col in []:
        value = f'{value:,.1f}'

    # else 0 decimal
    else: value = f'{value:,.0f}' 

    # For these columns, put in parentheses if negative
    parenthesis_cols = [
        'Daily Change', 'Daily % Change',
        'Unrealized Return', 'Unrealized % Return',
    ]
    if is_neg and col in parenthesis_cols:
        # Remove negative sign
        value = str(value)[1:]
        value = f'({value})'

    return value


# Helper to generate nice daily_table for HTML
# Puts df into HTML
def add_daily_table_to_html(tag, text, daily_table, col_order, color_cols, subpage_paths, stock_x):

    # Add columns
    with tag('table', style='max-width: 230px'):
        with tag('tr'):
            # Add col for stock names
            with tag('th'):
                text('')
            # Add cols for each column
            for col in daily_table.columns:
                with tag('th'):
                    with tag('center'):
                        text(col)
            # Add col for stock names
            with tag('th'):
                text('')
        # Add rows
        for stock in daily_table.index:
            with tag('tr'):
                # First col is href: stock name => subpage
                # make it bold
                with tag('th'):
                    with tag('a', href=subpage_paths[stock], style='font-weight: bold;'):
                        with tag('div', style='text-align: right;'):
                            text(stock)
                # Populate the rest of the cols
                for col in daily_table.columns:
                    with tag('td'):
                        value = daily_table.loc[stock, col]
                        # Check if negative
                        if type(value) != str: is_neg = value < 0
                        
                        # Do some formatting/rounding
                        value = make_nice_string(value, col)

                        # Include color
                        if col in color_cols['cols']:
                            if is_neg:
                                with tag('center'):
                                    with tag('div', klass='neg'):
                                        text(value)
                            else:
                                with tag('center'):
                                    with tag('div', klass='pos'):
                                        text(value)
                        else:
                            # Center value in cell
                            with tag('center'):
                                text(value)
                # Last col is href: stock name => Yahoo Finance
                with tag('th'):
                    if stock == 'Portfolio': link = subpage_paths[stock]
                    else: link = f'https://finance.yahoo.com/quote/{stock}/'
                    with tag('a', href=link, style='font-weight: bold;'):
                        with tag('div', style='text-align: left;'):
                            text(stock)

        # Go through each stock in daily_table and if stock == stock_x, put border
        # around the entire row
        for stock in daily_table.index:
            if stock == stock_x:
                with tag('style'):
                    text(f'tr:nth-child({daily_table.index.get_loc(stock)+2}) {{border: 4px solid black;}}')
        # Put vertical border before col
        for i, col in enumerate(col_order):
            if col in ['Daily % Change', 'Unrealized % Return', 'Weekly Vol', 'Qty', 'Trail PE']:
                with tag('style'):
                    text(f'th:nth-child({i+2}), td:nth-child({i+2}) {{border-left: 3px solid black;}}')
        # Put vertical border after col
        for i, col in enumerate(col_order):
            if col in ['Unrealized Return', 'Total Value']:
                with tag('style'):
                    text(f'th:nth-child({i+2}), td:nth-child({i+2}) {{border-right: 3px solid black;}}')


# Create subpages for each stock
def generate_html_page_stock(stock, page_save_path, daily_table, col_order, col_defs, color_cols, table_styles, subpage_paths):
    current_path = os.path.dirname(os.path.realpath(__file__))
    doc, tag, text = Doc().tagtext()

    with tag('html'):
        with tag('head'):
            with tag('center'):
                with tag('title'):
                    text(f'{stock}')
            with tag('style'):
                for style in table_styles:
                    text(style)
                text('ul {list-style-type: none;}')
                text('img {max-width: 1300px;}')
                # Make br size smaller
                text('br {line-height: 0.1;}')

                with tag('style'):
                    # Use style.css file
                    with open(os.path.join(current_path, 'style.css'), 'r') as f:
                        text(f.read())

        with tag('body'):
            with tag('center'):
                # Title 
                with tag('h1'):
                    with tag('u'):
                        text(f'{stock}')


                with tag('center'):
                    # Daily table ==========================
                    daily_table = daily_table[col_order]
                    add_daily_table_to_html(tag, text, daily_table, col_order, color_cols, subpage_paths, stock)

                with tag('br'):
                    pass
                with tag('br'):
                    pass
                with tag('br'):
                    pass
                with tag('br'):
                    pass

                # Trend plot
                trend_plot_path = os.path.join(current_path, 'fportfolio', 'images', f'{stock}.png')
                with tag('img', src=trend_plot_path, style='max-width: 1300px; width: 90%;'):
                    pass
    
    result = doc.getvalue() 
    with open(page_save_path, 'w') as f:
        f.write(result)


# Create simple HTML page w/ an image
def generate_html_page_simple(page_save_path, image_filename, title=''):
    doc, tag, text = Doc().tagtext()

    with tag('html'):
        with tag('head'):
            with tag('center'):
                with tag('title'):
                    text(title)

            with tag('style'):
                text('table {border-collapse: collapse;}')
                text('table, th, td {border: 1px solid black;}')
                text('th, td {padding: 5px;}')
                text('th {text-align: left;}')
                text('tr:nth-child(1) {background-color: #f2f2f2;}')
                text('tr:hover {background-color: #f1f1f1;}')
        with tag('body'):
            with tag('center'):
                with tag('h1'):
                    text(title)
            with tag('center'):
                # Add href back to main page
                with tag('a', href='portfolio_analysis.html', style='font-size: 30px;'):
                    text('Back to Main Page')
            with tag('center'):
                with tag('img', src=image_filename, width='60%'):
                    pass

    result = doc.getvalue()
    with open(page_save_path, 'w') as f:
        f.write(result)



# Create HTML page
def generate_html_page(daily_table):

    current_date = datetime.datetime.now().strftime('%A %B {S}')
    current_date = current_date.format(S=add_suffix(datetime.datetime.now().day))
    current_path = os.path.dirname(os.path.realpath(__file__))

    # Columns to include in table
    col_order = [
        'Daily % Change', 'Daily Change', 
        'Unrealized % Return', 'Unrealized Return', 
        'β', 'SPY Corr', 
        'Weekly Vol', 'Monthly Vol', 
        'Trail PE', 'Fwd PE', 'Trail EPS', 'Fwd EPS', 'Year $ Div',
        'Qty', 'Purchase Price', 'Cur. Price', 
        '% Portfolio', 'Total Value'
        ]

    definitions = get_definitions()
    col_defs = definitions['col_defs']
    indicator_definitions = definitions['indicator_definitions']

    # Columns to include red/green
    color_cols = {
        'cols': [ # Columns to color
            'Daily % Change', 'Daily Change', 
            'Unrealized % Return', 'Unrealized Return'],
        'neg': '#e6e6e6', # light grey
        'pos': '#ccffcc' # light green

    }
    subpage_paths = {'Yahoo Finance': 'https://finance.yahoo.com/'} # label -> path
    # Table styles
    table_styles = [
        'table {border-collapse: collapse; max-width: 1300px}',
        'table {border: 2px solid black;}',
        'td {border: 1px solid black;}',
        'th {border: 1.5px solid black;}',
        'th, td {padding: 5px;}',
        'th {text-align: left;}',
        'tr:nth-child(1) {background-color: #f2f2f2;}', # Header row grey
        'tr:hover {background-color: #f1f1f1;}', # Hover row grey
        # Create new div types pos, neg
        f'.pos {{background-color: {color_cols["pos"]};}}',
        f'.neg {{background-color: {color_cols["neg"]};}}'
    ]

    # Step 1: Create simple subplages ==========================
    root = os.path.dirname(os.path.realpath(__file__))
    corr_matrix_path = os.path.join(root, 'fportfolio', 'images', 'corr_matrix.png')
    corr_matrix_page = os.path.join(root, 'fportfolio', 'pages', 'corr_matrix.html')
    generate_html_page_simple(corr_matrix_page, corr_matrix_path, 'Pearson Correlation Matrix')
    subpage_paths['Correlation Matrix'] = corr_matrix_page

    # Step 2: Create subpages for each stock
    for stock in daily_table.index:
        if stock == 'Portfolio': stock_page = os.path.join(root, 'fportfolio', 'pages', 'portfolio_analysis.html')
        else: stock_page = os.path.join(root, 'fportfolio', 'pages', f'{stock}.html')
        subpage_paths[stock] = stock_page
    for stock in tqdm(daily_table.index, desc='Creating subpages'):
        if stock == 'Portfolio': continue
        generate_html_page_stock(stock, subpage_paths[stock], daily_table, col_order, col_defs, color_cols, table_styles, subpage_paths)

    # Step 3: Create full page ==========================

    # Create HTML page
    doc, tag, text = Doc().tagtext()

    with tag('html'):
        # Head
        with tag('head'):
            with tag('center'):
                with tag('title'):
                    text(f'Portfolio Analysis {current_date}')
            with tag('style'):
                # Use style.css file
                with open(os.path.join(current_path, 'style.css'), 'r') as f:
                    text(f.read())

        # Body
        with tag('body'):
            with tag('center'):

                # First row!
                with tag('div', klass='columnContainer'):
                    # Market status
                    with tag('div', klass='columnLhugger'):
                        with tag('div', klass='text-box', id='market_status', style='font-size: 20px;'):
                            text('Market Status:')

                    # Show current datetime
                    with tag('div', klass='text-box', id='current_dt', style='font-size: 30px; font-weight: bold;'):
                        current_time = datetime.datetime.now().strftime('%A, %B %d at %-I:%M:%S %p')
                        # Add the time zone
                        current_time += ' ' + datetime.datetime.now().astimezone().tzname()
                        text(current_time)
                        
                    # Next update
                    with tag('div', klass='columnRhugger'):
                        with tag('div', klass='text-box', id='next_update', style='font-size: 20px;'):
                            text('Next Update:')

                # Second row!
                with tag('div', klass='columnContainer'):
                    # Insert buttons 
                    with tag('div', klass='columnLHugger'):
                        with tag('div', klass='columnContainer'):
                            # Insert button to refresh page
                            with tag('div', klass='columnLhugger'):
                                with tag('button', 
                                         klass='hover-button', 
                                         onclick='location.reload()',
                                         style='width: 75px;'
                                         ):
                                    text('Refresh Page')
                            # Button to fetch latest data
                            with tag('div', klass='columnLhugger_noflex'):
                                with tag('button', 
                                         klass='hover-button', 
                                         onclick='runAnalysis()',
                                         style='font-size: 20px;'
                                         ):
                                    text('Fetch Data')

                    # Insert log
                    with tag('div', id='log2'):
                        pass

                    # Insert links
                    with tag('div', klass='columnRhugger'):
                        with tag('div', klass='links'):
                            with tag('ul'):
                                for label in subpage_paths:
                                    # Skip stocks
                                    if label in daily_table.index: continue
                                    with tag('li'):
                                        with tag('a', href=subpage_paths[label]):
                                            text(label)


                # Add javascript function and button
                with tag('script'):

                    # Load javascript code from javascript.js
                    with open(os.path.join(current_path, 'javascript.js'), 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for line in lines:
                            text(line)

                # Little space
                with tag('br'):
                    pass

                # Daily table ==========================
                # Only take columns we want
                daily_table = daily_table[col_order]
                add_daily_table_to_html(tag, text, daily_table, col_order, color_cols, subpage_paths, 'Portfolio')

                # Little space
                with tag('br'):
                    pass
                with tag('br'):
                    pass
                with tag('br'):
                    pass

                # Add plots ==========================
                # Total Portfolio value ==========================

                plot_path = os.path.join(current_path, 'fportfolio', 'images', 'portfolio_overview.png')
                with tag('img', src=plot_path, width='95%'):
                    pass

                # Add horizontal line
                with tag('hr'):
                    pass

                # Use the row, col divs to make grid of plots
                # Make two columns of plots, make them in line with each other row-wise
                stocks = [s for s in daily_table.index if s != 'Portfolio']
                stocks_row_1 = stocks[::2]
                stocks_row_2 = stocks[1::2]
                with tag('div', klass='row'):
                    # Left column
                    with tag('div', klass='columnL'):
                        for stock in stocks_row_1:
                            with tag('center'):
                                # Stock name
                                with tag('h2'):
                                    with tag('u'):
                                        # Add link to subpage
                                        with tag('a', href=subpage_paths[stock]):
                                            text(stock)
                                # Add plot
                                plot_path = os.path.join(current_path, 'fportfolio', 'images', f'{stock}.png')
                                with tag('img', src=plot_path, width='98%'):
                                    pass
                    # Right column
                    with tag('div', klass='columnR'):
                        for stock in stocks_row_2:
                            with tag('center'):
                                # Stock name
                                with tag('h2'):
                                    with tag('u'):
                                        # Add link to subpage
                                        with tag('a', href=subpage_paths[stock]):
                                            text(stock)
                                # Add plot
                                plot_path = os.path.join(current_path, 'fportfolio', 'images', f'{stock}.png')
                                with tag('img', src=plot_path, width='98%'):
                                    pass

                # Add horizontal line
                with tag('hr'):
                    pass

                # Add indicator definitions
                with tag('h2'):
                    text('Column Definitions')
                for col in col_order:
                    with tag('li', style='font-size: 20px;'):
                        text(f'{col}: {col_defs[col]}')
                # Little space
                with tag('br'):
                    pass

                # Add horizontal line
                with tag('hr'):
                    pass

                # Add indicator definitions
                with tag('h2'):
                    text('Indicator Definitions')
                with tag('div', style='max-width: 800px'):
                    for indicator in indicator_definitions:
                        definition = indicator_definitions[indicator]
                        lines = definition.split('.\n')
                        with tag('li', style='font-size: 24px;'):
                            with tag('u'):
                                text(indicator)
                        with tag('ul', style='font-size: 20px;'):
                            for line in lines:
                                with tag('li'):
                                    text(line)

                # Add horizontal line
                with tag('hr'):
                    pass

                # Write datetime generated
                with tag('h2'):
                    text(f'Generated on {datetime.datetime.now().strftime("%A %m/%d/%Y %H:%M:%S")}')

    result = doc.getvalue()
    result = dehtml(result) # De-HTML the symbols &, <, >, >=, <=
    filepath = os.path.join(current_path, 'fportfolio', 'pages', 'portfolio_analysis.html')
    with open(filepath, 'w') as f:
        f.write(result)
    print(f'Saved HTML page')
