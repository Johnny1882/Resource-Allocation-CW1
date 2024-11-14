import numpy as np

# Define the DAG as an adjacency matrix (using Python 0-based indexing)
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

# Define processing times and due dates
processing_times = [3,10,2,2,5,2,14,5,6,5,5,2,3,3,5,6,6,6,2,3,2,3,14,5,18,10,2,3,6,2,10]
due_dates = [172,82,18,61,93,71,217,295,290,287,253,307,279,73,355,34, 233,77,88,122,71,181,340,141,209,217,256,144,307,329,269]

# Check if a schedule is feasible given precedence constraints
def is_feasible(schedule, G):
    for i, j in edges:
        if schedule.index(i+1) > schedule.index(j+1):
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
def tabu_search(x0, processing_times, due_dates, G, L=20, gamma=10, K=1000):
    tabu_list = []
    best_solution = x0[:]
    best_tardiness = total_tardiness(x0, processing_times, due_dates)
    
    current_solution = x0[:]
    for k in range(K):
        neighborhood = generate_neighborhood(current_solution)
        
        best_move = None
        best_move_tardiness = float('inf')
        
        for neighbor in neighborhood:
            if is_feasible(neighbor, G):
                i, j = find_swapped_jobs(current_solution, neighbor)
                
                if (i, j) not in tabu_list or total_tardiness(neighbor, processing_times, due_dates) < best_tardiness:
                    tardiness = total_tardiness(neighbor, processing_times, due_dates)
                    
                    if tardiness < best_move_tardiness:
                        best_move = neighbor
                        best_move_tardiness = tardiness
                        if tardiness < best_tardiness:
                            best_solution = neighbor
                            best_tardiness = tardiness
        
        # Update Tabu List
        if len(tabu_list) >= L:
            tabu_list.pop(0)
        tabu_list.append((i, j))
        
        if best_move:
            current_solution = best_move
        else:
            break

    return best_solution, best_tardiness





# Initial solution x0
x0 = [30,29,23,10,9,14,13,12,4,20,22,3,27,28,8,7,19,21,26,18,25,17,15,6,24,16,5,11,2,1,31]
print(len(x0), len(processing_times),len(due_dates))

# Run the tabu search with K=10, K=100, and K=1000 iterations and L=20
print("Testing Tabu Search with different values of K:")

for K in [10, 100, 1000]:
    best_solution, best_tardiness = tabu_search(x0, processing_times, due_dates, G, L=20, gamma=10, K=K)
    print(f"Best solution with K={K}: {best_solution}")
    print(f"Total tardiness with K={K}: {best_tardiness}\n")
