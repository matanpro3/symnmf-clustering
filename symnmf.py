import sys
import numpy as np
import symnmf

# Set the random seed at the beginning as required
np.random.seed(1234)

# A small epsilon for floating point comparisons
EPSILON = 1e-4

def print_matrix(matrix):
    """
    Prints a matrix with a specific format.
    Rounds each value to 4 decimal places.
    
    Args:
        matrix (list of list or list): The matrix or vector to print.
    """
    if not matrix:
        return

    # Check if it's a 1D list (vector) or a 2D list (matrix)
    is_matrix = isinstance(matrix[0], list)
    
    for row in matrix:
        if is_matrix:
            # Print a 2D matrix row
            print(','.join([f"{x:.4f}" for x in row]))
        else:
            # Print a 1D vector
            print(','.join([f"{x:.4f}" for x in matrix]))
            break

def load_data(file_path):
    """
    Loads data points from a file.
    
    Args:
        file_path (str): The path to the data file.
    
    Returns:
        list of list: The data points.
    """
    try:
        with open(file_path, 'r') as f:
            points = []
            for line in f:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                coords = [float(x.strip()) for x in line.split(',')]
                points.append(coords)
            return points
    except Exception as e:
        print("An Error Has Occurred")
        sys.exit(1)

def initialize_H(W, k):
    """
    Initialize H matrix using numpy random as per requirements.
    
    Args:
        W: The normalized similarity matrix
        k: Number of clusters
    
    Returns:
        list of list: Initialized H matrix
    """
    n = len(W)
    
    # Compute average of W
    m = np.mean(W)
    
    # Scale factor
    scale = 2 * np.sqrt(m / k)
    
    # Initialize H using np.random.uniform as required
    H = np.random.uniform(0, scale, (n, k))
    
    return H.tolist()

def main():
    """
    Main function to parse arguments and run the symNMF algorithm or
    matrix calculation based on the provided goal.
    """
    # Check for correct number of arguments
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("An Error Has Occurred")
        sys.exit(1)

    try:
        if len(sys.argv) == 4:
            k = int(sys.argv[1])
            goal = sys.argv[2]
            file_name = sys.argv[3]
        elif len(sys.argv) == 3:
            goal = sys.argv[1]
            file_name = sys.argv[2]
            k = None # k is not needed for sym, ddg, or norm
        else:
            print("An Error Has Occurred")
            sys.exit(1)
            
        points = load_data(file_name)

    except (ValueError, IndexError):
        print("An Error Has Occurred")
        sys.exit(1)

    # Perform the requested operation
    if goal == "sym":
        A = symnmf.sym(points)
        print_matrix(A)
    elif goal == "ddg":
        D = symnmf.ddg(points)
        # Convert the diagonal vector to a full matrix for printing, as per typical DDG representation
        ddg_matrix = np.diag(D).tolist()
        print_matrix(ddg_matrix)
    elif goal == "norm":
        W = symnmf.norm(points)
        print_matrix(W)
    elif goal == "symnmf":
        if k is None or not (1 < k < len(points)):
            print("An Error Has Occurred")
            sys.exit(1)
        
        # Compute W first
        W = symnmf.norm(points)
        
        # Initialize H using numpy as required by assignment
        H_init = initialize_H(W, k)
        
        # Pass initialized H to the C function (which will do the iterative updates)
        H = symnmf.symnmf(W, H_init, k)
        print_matrix(H)
    else:
        print("An Error Has Occurred")
        sys.exit(1)

if __name__ == "__main__":
    main()