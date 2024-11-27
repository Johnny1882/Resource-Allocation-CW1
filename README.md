# Resource-Allocation-CW1
This project implements Least Cost Last (LSL) scheduling algorithm and Tabu search algorithm on single machine, precedence constraint, tardiness minimization scenarios.

## Requirements
To run the project, ensure you have the following Python packages installed:
- `numpy`
- `collections`
- `matplotlib`

---

## Files and Structure
- **`LSL.py`**: Contains the implementation of the LSL algorithm.
  - Detailed printout results are available in the `printout/LSL.txt` file.
  - Run the file directly, and the scheduling results in the printout will be updated.

- **`Tabu.py`**: Contains the implementation of the Tabu Search algorithm. The first half of the main function uses parameters γ (gamma) = 10 and L = 20, testing the results of K = 10, 100, and 1000. The second half fixes a particular K value, varying across a range of γ (gamma) and L, creating a 3D plot of total tardiness over the two parameters.
  - Printout results for different parameters are available, stored in the `printout` folder.
    - `tabu_k=10.txt`: Results for K = 10.
    - `tabu_k=100.txt`: Results for K = 100.
    - `tabu_k=1000.txt`: Results for K = 1000.
  - Run the file directly, and the two halves will both be executed. All printout texts will be updated, and a 3D plot will be drawn. User can vary K value of the second half (line 161, currently set to K=1000), as well as changing the range of γ (gamma) and L to make plot of (lines 159, 160).

---