import simpy
import math
import matplotlib.pyplot as plt
from matplotlib import pylab
from pylab import *
import random

class MMcQueue:
    def __init__(self, env, arrival_rate, service_rate, num_servers):
        # Initialize the MMcQueue class
        self.env = env
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.num_servers = num_servers
        # Create a queue for customers
        self.queue = simpy.Store(env)
        # Create resources for servers
        self.servers = [simpy.Resource(env, capacity=1) for _ in range(num_servers)]
        # Initialize metrics variables
        self.total_wait_time = 0
        self.total_customers_served = 0
        self.total_queue_length = 0
        self.total_time = 0
        self.max_queue_length = 0
        self.total_system_time = 0
        self.customers_in_system = [0]
        # Start the server processes
        for i in range(num_servers):
            self.env.process(self.server_process())

    def arrival_process(self):
        # Process for customer arrivals
        customer_id = 0
        while True:
            # Generate customer arrivals based on exponential distribution
            yield self.env.timeout(random.expovariate(self.arrival_rate))
            customer_id += 1
            print(f"Customer {customer_id} arrives at time {self.env.now}")
            # Start processing the arrived customer
            self.env.process(self.customer(customer_id))
            # Update queue length and time metrics
            self.total_queue_length += len(self.queue.items)
            self.total_time += self.env.now
            self.max_queue_length = max(self.max_queue_length, len(self.queue.items))
            self.customers_in_system.append(len(self.queue.items) + 1)

    def customer(self, customer_id):
        # Process for each customer
        arrival_time = self.env.now
        print(f"Customer {customer_id} enters the queue at time {self.env.now}. Queue length: {len(self.queue.items) + 1}")
        yield self.queue.put((arrival_time, customer_id))

    def server_process(self):
        # Process for server operations
        while True:
            # Get the next customer from the queue
            arrival_time, customer_id = yield self.queue.get()
            server_index = self.get_available_server()
            # Request server for processing
            with self.servers[server_index].request() as req:
                yield req
                print(f"Customer {customer_id} starts service at time {self.env.now} with Server {server_index + 1}. Queue length: {len(self.queue.items)}")
                # Generate service time based on exponential distribution
                service_time = random.expovariate(self.service_rate)
                yield self.env.timeout(service_time)
                # Update metrics after service completion
                self.customers_in_system.append(len(self.queue.items)) 
                self.total_system_time += self.env.now - arrival_time
                self.total_wait_time += self.env.now - arrival_time - service_time
                self.total_customers_served += 1
                self.total_queue_length += len(self.queue.items)
                self.total_time += self.env.now
                self.max_queue_length = max(self.max_queue_length, len(self.queue.items))
                print(f"Customer {customer_id} finishes service at time {self.env.now}.")

    def get_available_server(self):
        # Find an available server
        for i in range(self.num_servers):
            if self.servers[i].count == 0:
                return i
        return None  # If all servers are busy, return None

def calculate_performance_metrics(c, l, m):
    # Calculate theoretical performance metrics
    r = l / (c * m)
    if r >= 1:
        raise ValueError("The ratio of arrival rate to service rate must not be 1 or more when the number of servers is taken into account. The ratio of arrival rate to service is calculated by the following formula: {Number_of_servers / (arrival_rate * service_rate)}")
    p0 = 1 / (sum([(c * r) ** n / math.factorial(n) for n in range(c)]) + ((c * r) ** c) / (math.factorial(c) * (1 - r)))
    N = c * r + ((((c * r) ** c) * r) / (math.factorial(c) * ((1 - r) ** 2))) * p0
    T = 1 / m + (((c * r) ** c) / (math.factorial(c) * (c * m) * ((1 - r) ** 2))) * p0
    Nq = ((((c * r) ** c) * r) / (math.factorial(c) * ((1 - r) ** 2))) * p0
    Tq = Nq / arrival_rate
    return N, T, Nq, Tq

def simulate_queue(arrival_rate, service_rate, num_servers, sim_time, idx):   
    # Ensure idx is a number
    idx = float(idx)

    env = simpy.Environment()
    queue = MMcQueue(env, arrival_rate, service_rate, num_servers)
    env.process(queue.arrival_process())
    env.run(until=sim_time)
    # Calculate simulation metrics
    avg_wait_time = queue.total_wait_time / queue.total_customers_served
    avg_queue_length = queue.total_queue_length / queue.total_time
    avg_system_time = queue.total_system_time / queue.total_customers_served
    avg_customers_in_system = sum(queue.customers_in_system) / len(queue.customers_in_system)
    
    if(avg_wait_time < 0):
        avg_wait_time = 0

    # Print simulation metrics    
    print("The Simulation Values are:")
    print("     Average System Time:", avg_system_time)
    print("     Average Wait Time in the Queue:", avg_wait_time)
    print("     Average Queue Length:", avg_queue_length)
    print("     Maximum Queue Length:", queue.max_queue_length)
    print("     Total Customers Served:", queue.total_customers_served)
    print("     Average Customers in System:", avg_customers_in_system)
    print("     Total Time:", queue.total_time)

    # Create a prefix for the output files
    prefix = "{:.2f},".format((idx + 1) * 0.05)

    # Write the simulation values to files
    with open('T_R.txt', 'a') as t_r_file:
        t_r_file.write(prefix + str(avg_system_time) + ',')

    with open('TQ_R.txt', 'a') as tq_r_file:
        tq_r_file.write(prefix + str(avg_wait_time) + ',')

    with open('NQ_R.txt', 'a') as nq_r_file:
        nq_r_file.write(prefix + str(avg_queue_length) + ',')

    with open('N_R.txt', 'a') as n_r_file:
        n_r_file.write(prefix + str(queue.total_customers_served) + ',')

    with open('NQMAX_R.txt', 'a') as nqmax_r_file:
        nqmax_r_file.write(prefix + str(queue.max_queue_length) + '\n')

    return avg_system_time, avg_wait_time, avg_queue_length, queue.max_queue_length, queue.total_customers_served, avg_customers_in_system, queue.total_time

# Initialize lists to store the results of each simulation run
avg_system_times = []
avg_wait_times = []
avg_queue_lengths = []
max_queue_lengths = []
total_customers_served = []
avg_customers_in_systems = []
total_times = []

# Define list of mean arriving time intervals
l_values = [20.00, 10.00, 6.66, 5.00, 4.00, 3.33, 2.86, 2.50, 2.22, 2.00, 1.82, 1.66, 1.54, 1.43, 1.33, 1.25, 1.18, 1.11, 1.05]  

# Inputs
print("#Have in mind that the ratio of arrival rate to service rate must not be 1 or more when the number of servers is taken into account.#\n#The ratio of arrival rate to service is calculated by the following formula: {Number_of_servers / (arrival_rate * service_rate)}#\n#The arrival rates are the following: [20.00, 10.00, 6.66, 5.00, 4.00, 3.33, 2.86, 2.50, 2.22, 2.00, 1.82, 1.66, 1.54, 1.43, 1.33, 1.25, 1.18, 1.11, 1.05]#\n")
service_rate = float(input("Enter service rate (customers per time unit): "))  
num_servers = int(input("Enter number of servers: "))  
sim_time = int(input("Enter simulation time: "))  

# Clear the contents of the output files at the start of the program
with open('T_R.txt', 'w') as t_r_file, \
     open('TQ_R.txt', 'w') as tq_r_file, \
     open('N_R.txt', 'w') as n_r_file, \
     open('NQ_R.txt', 'w') as nq_r_file, \
     open('NQMAX_R.txt', 'w') as nqmax_r_file:
    pass  # Just opening the files to clear them

# Run the simulation for each arrival rate in l_values
idx = 0
for arrival_rate in l_values:
    # Iterate over each arrival rate in the list
    print(f"\nRunning simulation for arrival rate: {arrival_rate}")
    # Run simulation for the current arrival rate
    results = simulate_queue(arrival_rate, service_rate, num_servers, sim_time, idx)
    idx += 1 
    # Record simulation results
    avg_system_times.append(results[0])
    avg_wait_times.append(results[1])
    avg_queue_lengths.append(results[2])
    max_queue_lengths.append(results[3])
    total_customers_served.append(results[4])
    avg_customers_in_systems.append(results[5])
    total_times.append(results[6])

    # Calculate theoretical performance metrics
    N, T, Nq, Tq = calculate_performance_metrics(num_servers, arrival_rate, service_rate)
    
    # Ensure non-negative theoretical wait time
    if(Tq<0):
        Tq=0

    print("\nThe Theoretical Values are:")
    print("     Average System Time:", T)
    print("     Average Wait Time in the Queue:", Tq)   
    print("     Average Queue Length:", Nq)
    print("     Total Customers Served:", N)

    # Write theoretical values to files
    with open('T_R.txt', 'a') as f:
        f.write(str(T) + '\n')

    with open('TQ_R.txt', 'a') as f:
        f.write(str(Tq) + '\n')

    with open('NQ_R.txt', 'a') as f:
        f.write(str(Nq) + '\n')

    with open('N_R.txt', 'a') as f:
        f.write(str(N) + '\n')

# Compute the average of the results
avg_system_time = sum(avg_system_times) / len(avg_system_times)
avg_wait_time = sum(avg_wait_times) / len(avg_wait_times)
avg_queue_length = sum(avg_queue_lengths) / len(avg_queue_lengths)
max_queue_length = max(max_queue_lengths)
avg_customers_served = sum(total_customers_served) / len(total_customers_served)
avg_customers_in_system = sum(avg_customers_in_systems) / len(avg_customers_in_systems)
avg_total_time = sum(total_times) / len(total_times)

# Print the average of the results
print("\nThe Average Simulation Values are:")
print("     Average System Time:", avg_system_time)
print("     Average Wait Time in the Queue:", avg_wait_time)
print("     Average Queue Length:", avg_queue_length)
print("     Maximum Queue Length:", max_queue_length)
print("     Average Total Customers Served:", avg_customers_served)
print("     Average Customers in System:", avg_customers_in_system)
print("     Average Total Time:", avg_total_time)
print("     Total Customers Served:", sum(total_customers_served))

# Define a function to plot data from a file
def plot_data(filename, xlabel, ylabel, title):
	data = np.loadtxt(filename, delimiter=',')
	x = data[:, 0]
	if data.shape[1] == 2:  # Only simulation data available
		y_simulation = data[:, 1]
		plt.plot(x, y_simulation, label='Simulation')
	else:  # Both theoretical and simulation data available
		y_theoretical = data[:, 1]
		y_simulation = data[:, 2]
		plt.plot(x, y_theoretical, label='Theoretical')
		plt.plot(x, y_simulation, label='Simulation')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.legend()
	plt.grid(True)
   
# Define a function to create custom figure labels
def set_figure_label(plot_count, total_plots):
	fig = pylab.gcf()
	fig.canvas.manager.set_window_title(f"plot {plot_count}/{total_plots}")   
   
# Define the total plots and the plot count
total_plots=5
plot_count=0   
   
# Plot N_R.txt
plot_count+=1
plot_data('N_R.txt', 'R', 'N', 'N - R')
set_figure_label(plot_count, total_plots)
plt.show()

# Plot NQ_R.txt
plot_count+=1
plot_data('NQ_R.txt', 'R', 'NQ', 'NQ - R')
set_figure_label(plot_count, total_plots)
plt.show()

# Plot NQMAX_R.txt
plot_count+=1
plot_data('NQMAX_R.txt', 'R', 'NQMAX', 'NQMAX - R')
set_figure_label(plot_count, total_plots)
plt.show()

# Plot T_R.txt
plot_count+=1
plot_data('T_R.txt', 'R', 'T', 'T - R')
set_figure_label(plot_count, total_plots)
plt.show()

# Plot TQ_R.txt
plot_count+=1
plot_data('TQ_R.txt', 'R', 'TQ', 'TQ - R')
set_figure_label(plot_count, total_plots)
plt.show()
