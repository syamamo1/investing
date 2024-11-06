import asyncio, signal
from websockets.server import serve
from portfolio_analysis import main_wrapper as portfolio_analysis
from datetime import datetime


# Echo to console when server receives a message
def echo(message):
    response = f'Server recieved: {message}'
    print(response)
    return response


# Define the listener coroutine
async def listener(websocket):
    # Catch errors
    try:
        async for message in websocket:
            response = echo(message)

            # Send the response back to the client
            await websocket.send(response)

            # Run portfolio analysis
            if message == 'Update Portfolio':
                portfolio_analysis()
                # After updating table/graphs, refresh the page
                await websocket.send('Refresh Page')
    except Exception as e:
        print(f'[{datetime.now().strftime("%A %Y-%m-%d %H:%M:%S")}] An error occurred: {e}')


# Define the main coroutine
async def main():
    server = await serve(listener, 'localhost', 8080)

    print('Server started on ws://localhost:8080')
    print('=' * 40)

    # Create a future to keep the server running indefinitely
    stop_event = asyncio.Future()

    # Define a shutdown handler
    def shutdown():
        print('\n')
        print('=' * 40)
        print('Shutting down server gracefully...')
        stop_event.set_result(None)

    # Register the shutdown handler for different signals
    loop = asyncio.get_running_loop()
    
    loop.add_signal_handler(signal.SIGINT, shutdown) # (CTRL + C)
    loop.add_signal_handler(signal.SIGTSTP, shutdown) # (CTRL + Z)
    loop.add_signal_handler(signal.SIGQUIT, shutdown) # (CTRL + \)

    try:
        await stop_event  # Run until the stop_event is set
    finally:
        server.close()
        await server.wait_closed()
        print('Server has shut down')

if __name__ == '__main__':
    asyncio.run(main())