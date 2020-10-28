"""
    Created by Tom Vecchi for 2020 UQ Farmbot Thesis Project

    Plots the information contained in a given log file as a colour coded
    scatter plot. Plant size is on the y-axis, time of imaging is on the x-axis,
    and dots are red if dead plants were detected at the time that image was taken.
"""

# Visualises logs obtained of a given plant
import argparse
import logging
import matplotlib.pyplot as plt

# converts a timestamp to a monotonically increasing int so that it can be used on the plot
# as the x coordinate
def parse_timestamp_to_number(stamp):
    digits = ["0","1","2","3","4","5","6","7","8","9"]
    output = ""
    for c in stamp:
        if c in digits:
            output += c
    return float(float(output) / 1000000)

# Parses a .log file containing the logged health status and size information
def parse_log_file(filename):
    times = []
    areas = []
    statuses = []
    count = 0
    
    with open(filename, "r") as log:
        logging.info("Reading from file")
        while True:
            entry = log.readline()
            entry = entry.rstrip()

            # Check for EOF
            if not entry:
                if count > 1:
                    break
                else:
                    continue

            # Do string processing here to split out values of interest
            elements = entry.split(",")

            if len(elements) != 3: # Invalid entry
                logging.error("Invalid entry detected on line " + str(count + 1))
                continue
            else:
                times.append(parse_timestamp_to_number(elements[0]))
                areas.append(float(elements[1]))
                statuses.append((elements[2]))
                # Add more variables as needed

            count += 1

        logging.debug(str(statuses))
        # Get healthy, dead areas and dates
        return times, areas, statuses

# Generates plot of plant size over time, coloured based on health status
# red = dead plants were detected on that date, green = fully healthy
def plot_size_and_health(times, sizes, statuses):
    logging.info("Sorting data points by health status")
    healthy_areas = []
    healthy_times = []

    dead_areas = []
    dead_times = []

    for i in range(0, len(times)):
        if statuses[i] == "True": #Dead
            dead_times.append(times[i])
            dead_areas.append(sizes[i])
        else:
            healthy_times.append(times[i])
            healthy_areas.append(sizes[i])

    plt.plot(healthy_times, healthy_areas, 'go', dead_times, dead_areas, 'ro')
    plt.ylabel("Healthy area (pixels)")

    fig = plt.gcf()
    fig.canvas.set_window_title("Logged plant data from " + str(args.filename[0]))
    plt.show()
    return


# Main entry point
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='View plot of plant analytics log file')
    parser.add_argument('filename', metavar='filename', type=str, nargs='+',
                   help='Log file to test')

    args = parser.parse_args()
    times, sizes, statuses = parse_log_file(args.filename[0])
    plot_size_and_health(times, sizes, statuses)


    
        



 
