import time
import random

def generate_random_numbers():
    while True:
        number = random.randint(100, 999)  # Generate a 3-digit number
        yield f'{number};'  # Append a semicolon to the number
        time.sleep(0.1)
