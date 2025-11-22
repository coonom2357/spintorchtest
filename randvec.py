import random


def randvec(size, min_value=1, max_value=20):
    """
    Generate a random vector with random length and integer values.
    
    Args:
        size (int): Vector length
        min_value (int): Minimum integer value (default: 1)
        max_value (int): Maximum integer value (default: 20)
    
    Returns:
        list: A list representing the vector
    """
    vector = [random.randint(min_value, max_value) for _ in range(size)]
    return vector


def print_vector(vec):
    """Print a vector in a readable format."""  
    for element in vec:
        print(element)

if __name__ == "__main__":
    # Example usage
    print("Random vector (length 2 to 10, values 1-20):")
    vector = randvec(size = 3)
    print(f"Size: {len(vector)}")
    print("Random vector:")
    print(vector)

