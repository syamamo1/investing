# Hard coded positions
def get_positions():    
    # 1. Get Postitions (edge, IRA) ======================================
    # NOTE: Only specify IRA/edge for stocks where you own the stock in both
    positions = {
        'SPY': {
            'purchased_price': 400.00,
            'quantity': 100
        },

        'AAPL': {
            'purchased_price': 180.54,
            'quantity': 525
        },
        'MSFT': {
            'purchased_price': 404.20,
            'quantity': 234
        },
        'NVDA': {
            'purchased_price': 115.76,
            'quantity': 960
        }
    }

    # Combine edge and IRA positions: SPY, NDAQ
    # Ubiquity positions (401k)
    ubiquity = {
        'SWPPX': {
            'value': 1.0, # Updated 2024 11 01
            'unrealized_gain': 1.0,
            'quantity': 1.0
        }
    }
    return positions, ubiquity