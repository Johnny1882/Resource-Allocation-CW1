import numpy as np


# Define the weighted tardiness function
def total_weighted_tardiness(schedule, processing_times, due_dates):
    completion_time = 0
    weighted_tardiness = 0
    for job_num in schedule:
        completion_time += processing_times[job_num - 1]
        tardiness = max(0, completion_time - due_dates[job_num - 1])
        weighted_tardiness += tardiness * weights[job_num - 1]
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

def is_feasible(schedule, g):
    return True


# Modified Tabu search to display iterations
def tabu_search(x0, processing_times, due_dates, G, cost_func, L=2, gamma=100, K=3):
    tabu_list = []
    best_schedule = x0[:]
    best_cost = cost_func(x0, processing_times, due_dates)
    current_schedule = x0[:]
    current_cost = best_cost
    last_swap = 0

    for k in range(1, K + 1):     
        print("#########################")  
        print(f"new try iteration {k}")
        print("#########################")  
        # iterate through swap paris until find one that satisfy the condition
        for i in range(last_swap, last_swap + len(current_schedule) - 1):
            i = i % (len(current_schedule) - 1)
            print(f"switch {i} and {i+1}")
            neighbor_schedule = current_schedule[:]
            neighbor_schedule[i], neighbor_schedule[i + 1] = neighbor_schedule[i + 1], neighbor_schedule[i]

            # check for precedence
            if not is_feasible(neighbor_schedule, G):
                continue
            
            a, b = neighbor_schedule[i], neighbor_schedule[i+1]
            a, b = min(a, b), max(a, b)

            neighbor_cost = cost_func(neighbor_schedule, processing_times, due_dates)
            current_cost = cost_func(current_schedule, processing_times, due_dates)
            delta = current_cost - neighbor_cost

            condition1 = ((a, b) not in tabu_list and delta > -gamma)
            condition2 = (cost_func(neighbor_schedule, processing_times, due_dates) < best_cost)

            print(f"--- schedule: {neighbor_schedule}, cost: {neighbor_cost}, {condition1} {condition2}")
            if condition1 or condition2:
                if (a, b) not in tabu_list:
                    tabu_list.append((a, b))
                current_schedule = neighbor_schedule
                current_cost = cost_func(neighbor_schedule, processing_times, due_dates)
                last_swap = i+1
                break

        # Update Tabu List
        if len(tabu_list) >= L:
            tabu_list.pop(0)
        if current_cost < best_cost:
            best_cost = current_cost
            best_schedule = current_schedule
        print(f"current schedule: {current_schedule}")
        print(f"current cost: {current_cost}")

    return best_schedule, best_cost

##################################
# Tutorial 3.8
##################################

weights = [3,4,5,7]
processing_times = [16,11,4,8]
due_dates = [1, 2, 7, 9]
x0 = [4, 2, 1, 3]  # Initial schedule
best_solution, best_tardiness = tabu_search(
    x0, 
    processing_times, 
    due_dates, 
    0, # dummy
    total_weighted_tardiness, 
    L=2, 
    gamma=20, 
    K=4
)

print(f"Final solution: {best_solution}, Total Weighted Tardiness: {best_tardiness}")



##################################
# Tutorial 3.7
##################################

# weights = [14,12,1,12]
# processing_times = [10, 10, 13, 4]
# due_dates = [4, 2, 1, 12]
# x0 = [2, 1, 4, 3]  # Initial schedule

# best_solution, best_tardiness = tabu_search(
#     x0, 
#     processing_times, 
#     due_dates, 
#     0, # dummy
#     total_weighted_tardiness, 
#     L=2, 
#     gamma=100, 
#     K=3
# )

# print(f"Final solution: {best_solution}, Total Weighted Tardiness: {best_tardiness}")