// Get current time
function get_current_time(isFullTime) {
    if (isFullTime) { // Get in format Day name month date with suffix hh:mm:ss AM/PM and timezone
        // return new Date().toLocaleString('en-US', {weekday: 'long', month: 'long', day: '2-digit', hour12: true, hour: 'numeric', minute: 'numeric', second: 'numeric'});
        return new Date().toLocaleString('en-US', {weekday: 'long', month: 'long', day: '2-digit', hour12: true, hour: 'numeric', minute: 'numeric', second: 'numeric', timeZoneName: 'short'});
    } else { // Get in format hh:mm:ss
        return new Date().toLocaleTimeString('en-US', {hour12: true});
    }
} 

// Function to check if trading is open
function isTradingOpen() {
    // Get new york day and time
    const newYorkTime = new Date().toLocaleString('en-US', {timeZone: 'America/New_York'});
    const newYorkDate = new Date(newYorkTime);
    const day = newYorkDate.getDay();
    const hour = newYorkDate.getHours();
    const minute = newYorkDate.getMinutes();
    const second = newYorkDate.getSeconds();

    const isWeekend = (day == 0 || day == 6);
    const isTradingHours = (hour >= 9 && hour < 16) || (hour == 16 && minute == 0 && second == 0);
    return !isWeekend && isTradingHours;
}

// Update market status 
function update_isTradingOpen() {
    const marketStatus = isTradingOpen() ? "Open" : "Closed";
    const toDisplay = 'Market Status: ' + marketStatus;
    document.getElementById('market_status').innerHTML = toDisplay;
}

// Log to console and to HTML log container
const logContainer = document.getElementById('log2'); 
function log2(message, useFullTime=false) {
    if (useFullTime) {
        message = `[${get_current_time(isFullTime=true)}]: ${message}`;
    } else {
        message = `(${get_current_time(isFullTime=false)}) ${message}`;
    }
    console.log(message);
    const p = document.createElement('p');
    p.style.margin = '0px'; // change p height to 0px
    p.textContent = message;
    logContainer.appendChild(p);
    logContainer.scrollTop = logContainer.scrollHeight;
}

// Update current time
function update_current_time() {
    const current_time = get_current_time(isFullTime=true);
    document.getElementById('current_dt').innerHTML = current_time;
}

// WebSocket connection
var has_connection = false;
function connect() {
    socket = new WebSocket("ws://localhost:8080");

    // When connection is opened
    socket.onopen = function(event) {
        has_connection = true;
        socket.send("Client connected to server");
    };

    // When message received
    socket.onmessage = function(event) {
        const response = event.data;
        // portfolio analysis successfully finished
        if (response == "Refresh Page") {
            location.reload(true); // Reload page, ignore cache
            is_ready_to_run = true;
        }
        log2(response);
    };

    // When connection is closed
    socket.onclose = function(event) {
        if (has_connection) {
            log2(`WebSocket connection closed`);
            has_connection = false;
        } else {
            log2(`WebSocket connection failed`);
        }    
    };

    // When error occurs
    socket.onerror = function(event) {
        // Get error message
        console.error(event);
    };
}

// Attempt to reconnect every 5 seconds
// if connection is closed
// This way, we don't have to CTRL + R to refresh
// and reconnect to server. Less human input!
function reconnectIfClosed(ignoreMarketStatus=false) {
    // If connection is closed, try to reconnect
    if (!has_connection) {
        if (isTradingOpen() || ignoreMarketStatus) {
            connect();
        }
    }
}

// When fetch data button is clicked
function runAnalysis() {
    if (socket.readyState === WebSocket.OPEN) {
        if (is_ready_to_run) {
            socket.send("Update Portfolio");
            is_ready_to_run = false;
        }
        else {
            log2("Analysis already running");
        }
    }
    else {
        reconnectIfClosed(ignoreMarketStatus=true);
        is_ready_to_run = true;
    }
}


// Update next_update, the time remaining until
// we call runAnalysis
// dt_between_updates: how often to run runAnalysis (ms)
function updateNextUpdate(dt_between_updates) {
    // If market is closed, get how long until it opens
    if (!isTradingOpen()) {
        // Get the current date and time in New York
        const newYorkTime = new Date().toLocaleString('en-US', {timeZone: 'America/New_York'});
        const newYorkDate = new Date(newYorkTime);

        // Get the day of the week in New York
        const dayOfWeekNY = newYorkDate.toLocaleString('en-US', {weekday: 'long'});
        const hour = newYorkDate.getHours();
        const minute = newYorkDate.getMinutes();

        // Get desired date
        const desiredDate = new Date(newYorkDate);

        // Get the time until market opens
        if (['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'].includes(dayOfWeekNY)) {
            // Check if too early in the day (earlier than 9:30 AM)
            if (hour < 9 || (hour == 9 && minute < 30)) {
                desiredDate.setHours(9, 30, 0, 0);
            } else { // Too late in the day, go to next day
                desiredDate.setDate(desiredDate.getDate() + 1);
                desiredDate.setHours(9, 30, 0, 0);

                // If it's Friday, go to Monday
                if (dayOfWeekNY == 'Friday') {
                    desiredDate.setDate(desiredDate.getDate() + 2);
                }
            }

        // Weekends
        } else if (['Saturday', 'Sunday'].includes(dayOfWeekNY)) {
            desiredDate.setDate(desiredDate.getDate() + (dayOfWeekNY == 'Saturday' ? 2 : 1));
            desiredDate.setHours(9, 30, 0, 0);
        }

        // Calculate timeToOpen: between desired date and NY current date (ms)
        const timeToOpen = desiredDate - newYorkDate;
      
        // Calculate days, hours, minutes, seconds until market opens
        let days = Math.floor(timeToOpen / (1000 * 60 * 60 * 24));
        let hours = Math.floor((timeToOpen % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
        let minutes = Math.ceil((timeToOpen % (1000 * 60 * 60)) / (1000 * 60));

        // If minutes is 60, increment hours and set minutes to 0
        if (minutes == 60) {
            hours += 1;
            minutes = 0;
        }
        // If hours is 24, increment days and set hours to 0
        if (hours == 24) {
            days += 1;
            hours = 0;
        }

        // Display on div next_update the time until market opens
        let toDisplay = 'Market opens in: ';
        toDisplay += days > 0 ? days + 'd ' : '';
        toDisplay += hours > 0 ? hours + 'h ' : '';
        toDisplay += minutes + 'm';
        document.getElementById('next_update').innerHTML = toDisplay;
    }

    // If market is open, get how long until next update
    else {
        // First iteration, set updateTime to now + dt_between_updates
        if (updateTime == -1) {
            updateTime = new Date().getTime() + dt_between_updates;
        } 
        let distance = updateTime - new Date().getTime();
        // Not time yet, update time remaining
        if (distance > 0) {
            const minutes = Math.floor((distance % (1000 * 60 * 60)) / (1000 * 60));
            const seconds = Math.floor((distance % (1000 * 60)) / 1000);
            const toDisplay = 'Next update: ' + minutes + 'm ' + seconds + 's';
            document.getElementById('next_update').innerHTML = toDisplay;
        // Run analysis when distance is <= 0
        } else if (distance <= 0) {
            document.getElementById('next_update').innerHTML = "Next update: Now";
            runAnalysis();
            updateTime = new Date().getTime() + dt_between_updates; // Set next update time
        }
    }
}

// 1-second updates
function oneSecondUpdates() {
    update_current_time(); // current time at top of page
    update_isTradingOpen(); // display market status
    updateNextUpdate(dt_between_updates = 2 * 60 * 1000); // display next update time
}

// Call set interval functions
function call_setIntervals() {
    setInterval(oneSecondUpdates, 1 * 1000);
    
    // reload page if connection is closed and market is open
    setInterval(reconnectIfClosed, 10 * 1000); // every 10 seconds
}

// VARIABLES ======================================================
var updateTime = -1; // time we will run update
var is_ready_to_run = true; // so we don't run analysis while it's already running

// SET INTERVALS ======================================================
call_setIntervals();


// INITIALIZATION ======================================================
log2(`HTML Generated`, useFullTime=true);
connect();

