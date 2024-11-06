import os
import sys
from pathlib import Path

from portfolio_analysis import main as portfolio_analysis



def main():
    '''
    Setup directories
    Run analysis for first time to create page
    '''

    # Get the current working directory
    cwd = os.getcwd()

    # pfortfolio
    portfolio = Path(cwd) / 'fportfolio'
    for subdir in ['data', 'images', 'pages']:
        path = portfolio / subdir
        if not path.exists():
            path.mkdir(parents=True)
            print(f'Created directory: {path}')

    # portfolio_analysis
    portfolio_analysis(fast=False, stock_graphs=True)


if __name__ == '__main__':
    main()
    print(f'\nFinished setup!')
    print(f'Paste into Browser: \n\t{Path(os.getcwd()) / "fportfolio/pages/portfolio_analysis.html"}')