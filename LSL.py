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


# Define the Least Cost Last (LCL) scheduling function
def LCL_schedule(G, processing_times, due_dates):
    n = len(processing_times)
    scheduled = []
    unscheduled = set(range(n))
    completion_times = [0] * n  # Track completion times for each job
    partial_schedule_log = []  # Store partial schedules for reporting

    for step in range(n):
        # Identify jobs without successors in the unscheduled set
        candidate_jobs = {j for j in unscheduled if all(G[j, k] == 0 or k not in unscheduled for k in range(n))}
        
        # Calculate tardiness for each candidate job if scheduled last in the current set
        min_cost_job = None
        min_tardiness = float('inf')
        
        for job in candidate_jobs:
            # Assume the job is completed at time equal to sum of processing times of all unscheduled jobs
            completion_time = sum(processing_times[j] for j in unscheduled)
            tardiness = max(0, completion_time - due_dates[job])
            
            if tardiness < min_tardiness:
                min_tardiness = tardiness
                min_cost_job = job
                completion_times[job] = completion_time  # Record completion time for reporting

        # Schedule the selected job last in the current set and remove from unscheduled
        scheduled.insert(0, min_cost_job)
        unscheduled.remove(min_cost_job)
        
        # Log the partial schedule at specific iterations for reporting
        if step < 2 or step == n - 1 or step % 5 == 0:
            partial_schedule_log.append((step + 1, list(scheduled)))  # Log step and partial schedule
        
    return scheduled, completion_times, partial_schedule_log

# Run the LCL scheduling algorithm
final_schedule, completion_times, partial_schedule_log = LCL_schedule(G, processing_times, due_dates)

# Output the required iterations for the answer sheet
for iteration, partial_schedule in partial_schedule_log:
    print(f"Iteration {iteration}: Partial Schedule = {partial_schedule}")

tardiness = [max(0, completion_times[j] - due_dates[j]) for j in range(len(due_dates))]
max_tardiness = max(tardiness)
total_tardiness = sum(tardiness)

print(f"Final Schedule: {final_schedule}")
print(f"Completion Times: {completion_times}")

print(f"Maximum Tardiness: {max_tardiness}")
print(f"Total Tardiness: {total_tardiness}")