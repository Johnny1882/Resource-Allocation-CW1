import numpy as np
from collections import deque
import matplotlib.pyplot as plt

logger = open("change.txt", 'w')

def generate_initial_solution(G):
    in_degree = np.sum(G, axis=0)  # Calculate in-degree of each node
    queue = deque(i for i in range(in_degree.shape[1]) if in_degree[i]==0)    # Queue to store nodes with in-degree 0

    schedule = []
    while queue:
        node = queue.popleft()
        schedule.append(node + 1)  # Add job to the schedule (1-based index)

        # Decrease the in-degree of its neighbors
        for neighbor in range(len(G)):
            if G[node, neighbor]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

    return schedule

# Check if a schedule is feasible given precedence constraints
def is_feasible(schedule):
    schedule_index = {schedule[s]-1: s for s in range(len(schedule))}
    for i, j in edges:
        if schedule_index[i] > schedule_index[j]:
            return False
    return True

# Calculate the total tardiness for a given schedule
def total_tardiness(schedule, processing_times, due_dates):
    completion_time = 0
    tardiness = 0
    for job_num in schedule:
        completion_time += processing_times[job_num - 1]
        tardiness += max(0, completion_time - due_dates[job_num - 1])
    return tardiness

def total_weighted_tardiness(schedule, processing_times, due_dates):
    completion_time = 0
    weighted_tardiness = 0
    for job_num in schedule:
        completion_time += processing_times[job_num - 1]
        tardiness = max(0, completion_time - due_dates[job_num - 1])
        weighted_tardiness += tardiness * (job_num + 1)  # Example weight: (job index + 1)
    return weighted_tardiness

# Generate neighbors by swapping adjacent jobs cyclically
def generate_neighborhood(schedule):
    neighborhood = []
    for i in range(len(schedule) - 1):
        neighbor = schedule[:]
        neighbor[i], neighbor[i + 1] = neighbor[i + 1], neighbor[i]
        neighborhood.append(neighbor)
    return neighborhood

# Find the swapped jobs between two schedules
def find_swapped_jobs(schedule1, schedule2):
    for i in range(len(schedule1)):
        if schedule1[i] != schedule2[i]:
            return schedule1[i], schedule2[i]
    return None, None

# Main Tabu search function
def tabu_search(x0, processing_times, due_dates, cost_func, L=2, gamma=100, K=3):
    tabu_list = deque(maxlen=L)
    best_schedule = x0[:]
    best_cost = cost_func(x0, processing_times, due_dates)
    current_schedule = x0[:]
    current_cost = best_cost
    last_swap = 0

    for k in range(1, K + 1):     
        
        # iterate through swap pairs until find one that satisfy the condition
        for i in range(last_swap, last_swap + len(current_schedule) - 1):
            i = i % (len(current_schedule) - 1)
            # print(f"switch {i} and {i+1}")
            neighbor_schedule = current_schedule[:]
            neighbor_schedule[i], neighbor_schedule[i + 1] = neighbor_schedule[i + 1], neighbor_schedule[i]

            # check for precedence
            if not is_feasible(neighbor_schedule):
                continue
            
            a, b = neighbor_schedule[i], neighbor_schedule[i+1]
            a, b = min(a, b), max(a, b)

            neighbor_cost = cost_func(neighbor_schedule, processing_times, due_dates)
            current_cost = cost_func(current_schedule, processing_times, due_dates)
            delta = current_cost - neighbor_cost

            condition1 = ((a, b) not in tabu_list and delta > -gamma)
            condition2 = (neighbor_cost < best_cost)

            # print(f"--- schedule: {neighbor_schedule}, cost: {neighbor_cost}, {condition1} {condition2}")
            if condition1 or condition2:
                if (a, b) not in tabu_list:
                    tabu_list.append((a, b))
                current_schedule = neighbor_schedule
                current_cost = neighbor_cost
                last_swap = i+1
                break

        # Update Tabu List
        if current_cost < best_cost:
            best_cost = current_cost
            best_schedule = current_schedule

            logger.write(f"############## iteration {k} ################# \n")  
            logger.write(f"current cost: {current_cost}\n")
            logger.write(f"current schedule: {current_schedule} \n")
            logger.write(f"best cost: {best_cost} \n")
            logger.write(f"Tabu list: {tabu_list} \n")

    return best_schedule, best_cost


####################
# CW Test
####################
G = np.zeros((31, 31), dtype=np.bool_)
edges = [
    (0, 30), (1, 0), (2, 7), (3, 2), (4, 1), (5, 15), (6, 5), (7, 6), (8, 7), (9, 8),
    (10, 0), (11, 4), (12, 11), (13, 12), (16, 14), (14, 10), (15, 4), (16, 15),
    (17, 16), (18, 17), (19, 18), (20, 17), (21, 20), (22, 21), (23, 4), (24, 23),
    (25, 24), (26, 25), (27, 25), (28, 26), (28, 27), (29, 3), (29, 9), (29, 13),
    (29, 19), (29, 22), (29, 28)
]
for i, j in edges:
    G[i, j] = 1

# Define processing times and due dates
processing_times = [3,10,2,2,5,2,14,5,6,5,5,2,3,3,5,6,6,6,2,3,2,3,14,5,18,10,2,3,6,2,10]
due_dates = [172,82,18,61,93,71,217,295,290,287,253,307,279,73,355,34,233,77,88,122,71,181,340,141,209,217,256,144,307,329,269]
x0 = [30,29,23,10,9,14,13,12,4,20,22,3,27,28,8,7,19,21,26,18,25,17,15,6,24,16,5,11,2,1,31]

print("Testing Tabu Search with different values of K = 10, 100, 1000:")

best_solution, best_tardiness = tabu_search(x0, processing_times, due_dates, total_tardiness, L=20, gamma=10, K=1000)
# print(f"Best solution with K={K}: {best_solution}")
# print(f"Total tardiness with K={K}: {best_tardiness}\n")


####################
# Other Tests   
####################

# def is_valid_schedule(schedule, G):
#     # Check if the schedule respects the precedence constraints (DAG)
#     for i, j in edges:
#         if schedule.index(i + 1) > schedule.index(j + 1):
#             return False
#     return True

# # initial_solution = generate_initial_solution(G)
# initial_solution = [30, 29, 4, 3, 20, 10, 9, 8, 19, 14, 23, 7, 22, 21, 6, 18, 27, 13, 12, 28, 26, 17, 16, 25, 24, 5, 2, 15, 11, 1, 31]
# is_valid = is_valid_schedule(initial_solution, G)
# print("Is the initial solution valid?", is_valid)


##################################################################################################
######### Ploting Diagram
##################################################################################################

# Define the range of gamma and L values to test
gamma_values = range(1, 101, 5)  # Gamma values from 1 to 20
L_values = range(1, 101, 5)      # L values from 1 to 20

# Storage for results
results = []

# Iterate through all combinations of gamma and L
for gamma in gamma_values:
    for L in L_values:
        _, best_tardiness = tabu_search(x0, processing_times, due_dates, total_tardiness, L, gamma, K=1000)
        results.append((L, gamma, best_tardiness))

# Convert results to a 2D array for plotting
L_gamma_tardiness = np.array(results, dtype=int)
# Find the best combination and remove from all results
opt_index = np.argmin(L_gamma_tardiness[:, 2])
best_L, best_gamma, best_tardiness = results.pop(opt_index)
L_gamma_tardiness = np.array(results, dtype=int)

# Extract L, gamma, and tardiness values except for the best
L_values = L_gamma_tardiness[:, 0]
gamma_values = L_gamma_tardiness[:, 1]
tardiness_values = L_gamma_tardiness[:, 2]

# Plot the results in 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Create 3D scatter plot
scatter = ax.scatter(L_values, gamma_values, tardiness_values, c=tardiness_values, cmap='viridis', s=50)

# Highlight the best combination in red
ax.scatter([best_L], [best_gamma], [best_tardiness], color='red', label=f'Best Combination: L={best_L}, gamma={best_gamma}, tardiness={best_tardiness}', s=100, edgecolors='black')

# Add color bar and labels
fig.colorbar(scatter, ax=ax, label='Best Tardiness')
ax.set_title('Tardiness vs. L and Gamma (3D Plot)')
ax.set_xlabel('L (Tabu List Length)')
ax.set_ylabel('Gamma (Penalty Factor)')
ax.set_zlabel('Best Tardiness')
ax.legend()

plt.show()
print(f"Best Combination: L={int(best_L)}, Gamma={int(best_gamma)}, Best Tardiness={int(best_tardiness)}")

logger.close()