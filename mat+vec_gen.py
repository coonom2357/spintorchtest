import random


def gen_rand_mat_vec(min_size=2, max_size=10, min_value=1, max_value=20):
    """
    Generate a random matrix with random dimensions and integer values.
    
    Args:
        min_size (int): Minimum matrix dimension (default: 2)
        max_size (int): Maximum matrix dimension (default: 10)
        min_value (int): Minimum integer value (default: 1)
        max_value (int): Maximum integer value (default: 20)
    
    Returns:
        list: A 2D list representing the matrix
    """
    column = random.randint(min_size, max_size)
    rows = random.randint(min_size, max_size)
    
    matrix = [[random.randint(min_value, max_value) for _ in range(column)] for _ in range(rows)]
    vector = [random.randint(min_value, max_value) for _ in range(column)]
    return matrix, vector


def print_matrix(matrix):
    """Print a matrix in a readable format."""  
    for row in matrix:
        print(row)


if __name__ == "__main__":
    # Example usage
    print("Random matrix (2x2 to 10x10, values 1-20):")
    matrix, vector = gen_rand_mat_vec()
    print(f"Size: {len(matrix)}x{len(matrix[0])}")
    print_matrix(matrix)
    print("Random vector:")
    print(vector)

