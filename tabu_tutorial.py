import numpy as np

weights = [14, 12, 1, 12]
processing_times = [10, 10, 13, 4]
due_dates = [4, 2, 1, 12]
x0 = [2, 1, 4, 3]  # Initial schedule


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



# Modified Tabu search to display iterations
def tabu_search(x0, processing_times, due_dates, cost_func, L=2, gamma=100, K=3):
    tabu_list = []
    best_solution = x0[:]
    best_tardiness = cost_func(x0, processing_times, due_dates)
    current_solution = x0[:]

    print(f"Initial solution: {best_solution}, Tardiness: {best_tardiness}")
    
    for k in range(K + 1):
        # Generate all possible neighbors
        print("#################################")
        print(f"Iteration Number {k}")
        print("#################################")
        neighborhood = generate_neighborhood(current_solution)
        
        best_move = None
        best_move_tardiness = float('inf')
        
        for neighbor in neighborhood:
            i, j = find_swapped_jobs(current_solution, neighbor)
            neighbor_cost = cost_func(neighbor, processing_times, due_dates)
            current_cost = cost_func(current_solution, processing_times, due_dates)
            delta = current_cost - neighbor_cost
            print(f"Trying to switch jobs: {i, j}, Delta: {delta}")

            condition1 = ((i, j) not in tabu_list and delta < gamma)
            condition2 = (cost_func(neighbor, processing_times, due_dates) < best_tardiness)

            if condition1 or condition2:
                print(f"condition satisfied: {condition1, condition2}")
                # stop when the first feasible neighbor is found
                best_move = neighbor
                best_move_tardiness = cost_func(neighbor, processing_times, due_dates)

                break
                
                
                # if tardiness < best_move_tardiness:
                #     best_move = neighbor
                #     best_move_tardiness = tardiness

                #     if tardiness < best_tardiness:
                #         best_solution = neighbor
                #         best_tardiness = tardiness
            else:
                print(f"condition not satisfied: {condition1, condition2}")
        
        # Update Tabu List
        if len(tabu_list) >= L:
            tabu_list.pop(0)
        if best_move:
            i, j = find_swapped_jobs(current_solution, best_move)
            tabu_list.append((min(i, j), max(i, j)))
            current_solution = best_move
            
            # update best solution if needed
            if best_move_tardiness < best_tardiness:
                best_solution = best_move
                best_tardiness = best_move_tardiness
        else:
            break
        print(f"Tabu List,{tabu_list}")
        print(f"Iteration {k}: Current solution: {current_solution}, Tardiness: {best_move_tardiness}")
    
    return best_solution, best_tardiness



best_solution, best_tardiness = tabu_search(
    x0, 
    processing_times, 
    due_dates, 
    total_weighted_tardiness, 
    L=2, 
    gamma=100, 
    K=3
)

print(f"Final solution: {best_solution}, Total Weighted Tardiness: {best_tardiness}")