import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# Generate an feasible initial solution using topological sorting
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

# Main Tabu search function
def tabu_search(x0, processing_times, due_dates, cost_func, L=2, gamma=100, K=3, log=False):
    # Initialization
    tabu_list = deque(maxlen=L)
    best_schedule = x0[:]
    best_cost = cost_func(x0, processing_times, due_dates)
    current_schedule = x0[:]
    current_cost = best_cost
    last_swap = 0
    with open("printout/tabu_head.txt", "w") as f:
        f.write("") # clear old tabu head log
    # Iteration loop, each iteration swap two jobs in the schedule
    for k in range(1, K + 1):

        # Inner loop, iterate through swap paris until find one that satisfy the condition
        # Note: Iteration begin at index where last swapped happened, going from left to right
        # and stop immediately when a acceptable solution is found
        for i in range(last_swap, last_swap + len(current_schedule) - 1):
            i = i % (len(current_schedule) - 1)

            neighbor_schedule = current_schedule[:]
            neighbor_schedule[i], neighbor_schedule[i + 1] = neighbor_schedule[i + 1], neighbor_schedule[i]

            # check for new schedule staisfy precedence rules
            if not is_feasible(neighbor_schedule):
                continue

            # a, b are the jobs to be swapped, sort them to avoid duplicate tabu list
            a, b = neighbor_schedule[i], neighbor_schedule[i+1]
            a, b = min(a, b), max(a, b)

            neighbor_cost = cost_func(neighbor_schedule, processing_times, due_dates)
            current_cost = cost_func(current_schedule, processing_times, due_dates)
            delta = current_cost - neighbor_cost

            # two conditions to accept the new schedule
            condition1 = ((a, b) not in tabu_list and delta > -gamma)
            condition2 = (neighbor_cost < best_cost)

            # if either condition is satisfied, update record and go to next iteration
            if condition1 or condition2:
                if (a, b) not in tabu_list:
                    tabu_list.append((a, b))
                current_schedule = neighbor_schedule
                current_cost = neighbor_cost
                last_swap = i+1
                break
        if k <= 5:      # log tabu first 5 iterations
            with open("printout/tabu_head.txt", "a") as f:
                f.write(f"Iteration {k} schedule: "+str(current_schedule) + "\n")

        # Update best solution
        if current_cost < best_cost:
            best_cost = current_cost
            best_schedule = current_schedule
            if log:
                # Log the current state
                logger.write(f"---------- iteration {k} ----------\n")  
                logger.write(f"current cost: {current_cost}\n")
                logger.write(f"current schedule: {current_schedule} \n")
                logger.write(f"best cost: {best_cost} \n")
                logger.write(f"Tabu list: {tabu_list} \n")

    return best_schedule, best_cost


if __name__ == "__main__":
    # Define DAG, processing times, due dates and Initial schedule
    G = np.zeros((31, 31), dtype=int)
    edges = [
        (0, 30), (1, 0), (2, 7), (3, 2), (4, 1), (5, 15), (6, 5), (7, 6), (8, 7), (9, 8),
        (10, 0), (11, 4), (12, 11), (13, 12), (16, 14), (14, 10), (15, 4), (16, 15),
        (17, 16), (18, 17), (19, 18), (20, 17), (21, 20), (22, 21), (23, 4), (24, 23),
        (25, 24), (26, 25), (27, 25), (28, 26), (28, 27), (29, 3), (29, 9), (29, 13),
        (29, 19), (29, 22), (29, 28)
    ]
    for i, j in edges:
        G[i, j] = 1

    processing_times = [3,10,2,2,5,2,14,5,6,5,5,2,3,3,5,6,6,6,2,3,2,3,14,5,18,10,2,3,6,2,10]
    due_dates = [172,82,18,61,93,71,217,295,290,287,253,307,279,73,355,34,233,77,88,122,71,181,340,141,209,217,256,144,307,329,269]
    x0 = [30,29,23,10,9,14,13,12,4,20,22,3,27,28,8,7,19,21,26,18,25,17,15,6,24,16,5,11,2,1,31]

    ####################
    # 1. Testing multiple K values
    ####################

    for K in [10, 100, 1000]:
        logger = open(f"printout/tabu_k={K}.txt", 'w')
        logger.write(f"############## Running Tabu List with K = {K} ################# \n")
        best_solution, best_tardiness = tabu_search(x0, processing_times, due_dates, total_tardiness, L=20, gamma=10, K=K, log=True)

        logger.write("############## Final Results ################# \n")
        logger.write(f"Best solution with K={K}: {best_solution} \n")
        logger.write(f"Total tardiness with K={K}: {best_tardiness}\n")

    ##################################################################################################
    # 2. Ploting Diagrams over L and gamma
    ##################################################################################################

    gamma_values = range(1, 201, 5)  
    L_values = range(1, 101, 5)     
    K = 100
    # Storage for results
    results = []
    best_num, best_tardiness, best_solution = 0, None, None
    comb_num = 0
    for gamma in gamma_values:
        for L in L_values:
            schedule, tardiness = tabu_search(x0, processing_times, due_dates, total_tardiness, L, gamma, K)
            if best_tardiness is None or tardiness < best_tardiness:
                best_tardiness = tardiness
                best_solution = schedule
                best_num = comb_num
            results.append((L, gamma, tardiness))
            comb_num += 1
    best_L, best_gamma, best_tardiness = results.pop(best_num)
    # Convert results to a 2D array for plotting
    L_gamma_tardiness = np.array(results, dtype=int)

    # Extract L, gamma, and tardiness values except for the best
    L_values = L_gamma_tardiness[:, 0]
    gamma_values = L_gamma_tardiness[:, 1]
    tardiness_values = L_gamma_tardiness[:, 2]

    print(f"Best Combination: L={int(best_L)}, Gamma={int(best_gamma)}, Best Tardiness={int(best_tardiness)}")
    print(f"Optimal schedule:", best_solution)

    # Plot the results in 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Highlight the best combination in red
    ax.scatter([best_L], [best_gamma], [best_tardiness], color='red', label=f'Best Combination: L={best_L}, gamma={best_gamma}, tardiness={best_tardiness}', s=100, edgecolors='black')

    # Create 3D scatter plot
    scatter = ax.scatter(L_values, gamma_values, tardiness_values, c=tardiness_values, cmap='viridis', s=50)
    # Add color bar and labels
    fig.colorbar(scatter, ax=ax, label='Best Tardiness')
    ax.set_title(f'Tardiness vs. L and Gamma [K = {K}]')
    ax.set_xlabel('L (Tabu List Length)')
    ax.set_ylabel('Gamma (Penalty Factor)')
    ax.set_zlabel('Best Tardiness')
    ax.legend()

    plt.show()
