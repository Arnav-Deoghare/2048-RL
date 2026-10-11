import numpy as np

board_size = 4

# np.random.seed(0)  # Optional: for reproducibility


def initialise_grid():
    grid = np.zeros((board_size, board_size), dtype=int)

    positions = np.random.choice(
        board_size * board_size,
        size=2,
        replace=False
    )

    for pos in positions:
        row, col = divmod(pos, board_size)
        grid[row, col] = 2

    return grid

def spawn_tile(grid):
    empty_pos = np.argwhere(grid == 0)

    if empty_pos.size == 0:
        return

    new_pos = empty_pos[np.random.choice(empty_pos.shape[0])]
    k = np.random.choice([2, 4], p=[0.9, 0.1])
    grid[new_pos[0], new_pos[1]] = k

def process_line(line, direction):
    modified_line = line[line != 0]
    combined_line = []
    line_score = 0

    i = 0

    while i < modified_line.size:
        if (
            i + 1 < modified_line.size
            and modified_line[i] == modified_line[i + 1]
        ):
            merged_value = modified_line[i] + modified_line[i + 1]

            combined_line.append(merged_value)
            line_score += merged_value

            i += 2
        else:
            combined_line.append(modified_line[i])
            i += 1

    needed = board_size - len(combined_line)

    for _ in range(needed):
        if direction in ("d", "s"):
            combined_line.insert(0, 0)
        else:
            combined_line.append(0)

    return combined_line, line_score

def move_horizontal(grid, direction):
    move_score = 0

    for row in range(board_size):
        grid[row], line_score = process_line(grid[row], direction)
        move_score += line_score

    return grid, move_score

def move_vertical(grid, direction):
    move_score = 0

    for column in range(board_size):
        grid[:, column], line_score = process_line(
            grid[:, column],
            direction
        )
        move_score += line_score

    return grid, move_score

def move(grid, direction):
    if direction in ("a", "d"):
        return move_horizontal(grid, direction)

    elif direction in ("w", "s"):
        return move_vertical(grid, direction)

    else:
        raise ValueError("Invalid direction")

    
def is_game_over(grid):
    # If there is an empty cell, the game is not over
    if np.any(grid == 0):
        return False

    # Check horizontal neighbors
    for row in range(board_size):
        for col in range(board_size - 1):
            if grid[row, col] == grid[row, col + 1]:
                return False

    # Check vertical neighbors
    for row in range(board_size - 1):
        for col in range(board_size):
            if grid[row, col] == grid[row + 1, col]:
                return False

    # Board is full and no merges are possible
    return True

def main():
    grid = initialise_grid()
    print(grid)
    global_score = 0

    while True:
        inp = input("next move (W/A/S/D): ").strip().lower()

        if inp in ("w", "a", "s", "d"):
            before = grid.copy()

            grid, move_score = move(grid, inp)
            global_score += move_score

            # Only spawn if the move actually changed the board
            if not np.array_equal(before, grid):
                spawn_tile(grid)

            print(grid)
            print("Score:", global_score)

            if is_game_over(grid):
                print("GAME OVER")
                break

        else:
            print("Invalid move. Please enter W, A, S, or D.")
if __name__ == "__main__":
    main()

# # Test
# grid = np.array([
#     [2, 2, 2, 0],
#     [2, 2, 2, 2],
#     [4, 4, 8, 8],
#     [0, 0, 0, 0]
# ])

# print("Original:")
# print(grid)

# grid = move(grid, "w")

# print("\nMoved up:")
# print(grid)