import numpy as np
logger = open("LSL.txt", 'w')

def LCL_schedule(G, processing_times, due_dates):
    """
    Schedule jobs using the Longest Chain Last (LCL) scheduling algorithm.
    """
    # Initialization
    n = len(processing_times) # iteration required
    scheduled = [] # final schedule, update each iteration
    completion_times = np.zeros(n, dtype=int)  # Track completion times for each job
    completion_remain = np.sum(processing_times)   # Sum of remaining unscheduled jobs' processing times
    partial_schedule_log = []  # Store partial schedules for reporting

    for step in range(n):
        # Identify jobs without successors in the unscheduled set
        candidate_jobs = filter(lambda j: j not in scheduled and all(not G[j, k] or k in scheduled for k in range(n)), range(n))
        
        # Calculate tardiness for each candidate job if scheduled last in the current set
        min_cost_job = None
        min_tardiness = float('inf')
        
        # iterate through all candidate jobs, identify the one with the minimum tardiness
        for job in candidate_jobs:
            tardiness = max(0, completion_remain - due_dates[job])
            if tardiness < min_tardiness:
                min_tardiness = tardiness
                min_cost_job = job
                completion_times[job] = completion_remain 

        # Schedule the selected job last in the current set and remove from unscheduled
        scheduled.insert(0, min_cost_job)
        completion_remain -= processing_times[min_cost_job]
        
        # Log the partial schedule at specific iterations for reporting
        if step < 2 or step == n - 1 or step % 5 == 0:
            partial_schedule_log.append((step + 1, list(scheduled)))
        
    return scheduled, completion_times, partial_schedule_log

if __name__ == "__main__":
    # Define the DAG, processing time and due dates
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
    processing_times = np.array([3,10,2,2,5,2,14,5,6,5,5,2,3,3,5,6,6,6,2,3,2,3,14,5,18,10,2,3,6,2,10], dtype=int)
    due_dates = np.array([172,82,18,61,93,71,217,295,290,287,253,307,279,73,355,34, 233,77,88,122,71,181,340,141,209,217,256,144,307,329,269], dtype=int)


    # Run the LCL scheduling algorithm and output the results
    final_schedule, completion_times, partial_schedule_log = LCL_schedule(G, processing_times, due_dates)

    # Output the required iterations for the answer sheet
    for iteration, partial_schedule in partial_schedule_log:
        logger.write(f"Iteration {iteration}: Partial Schedule = {partial_schedule}\n")

    tardiness = np.maximum(0, completion_times-due_dates)
    max_tardiness = np.max(tardiness)
    total_tardiness = np.sum(tardiness)

    logger.write(f"####### Final Results: ######## \n")
    logger.write(f"Final Schedule: {final_schedule} \n")
    logger.write(f"Completion Times: {completion_times}\n")
    logger.write(f"Maximum Tardiness: {max_tardiness}\n")
    logger.write(f"Total Tardiness: {total_tardiness}\n")