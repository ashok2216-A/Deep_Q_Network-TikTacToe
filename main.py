import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from IPython.display import clear_output


# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Constants
WIDTH, HEIGHT = 600, 600
GRID_SIZE = 3
CELL_SIZE = WIDTH // GRID_SIZE
LINE_COLOR = (0, 0, 0)
BG_COLOR = (240, 240, 240)
X_COLOR = (66, 66, 245)
O_COLOR = (245, 66, 66)
LINE_WIDTH = 5
FONT_SIZE = 80

# DQN parameters
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.9
EXPLORATION_RATE = 0.2
EXPLORATION_DECAY = 0.995
MIN_EXPLORATION_RATE = 0.01
MEMORY_SIZE = 1000
BATCH_SIZE = 32
EPISODES = 10000
RENDER_INTERVAL = 5  # Render the game every 5 episodes

# Initialize memory
memory = []

# Functions
def get_state(board):
    return tuple(board.flatten())

def get_possible_actions(board):
    return [i for i in range(9) if board[i // 3, i % 3] == 0]

def check_winner(board):
    for row in board:
        if abs(sum(row)) == 3:
            return row[0]
    for col in board.T:
        if abs(sum(col)) == 3:
            return col[0]
    if abs(sum(board.diagonal())) == 3:
        return board[0, 0]
    if abs(sum(np.fliplr(board).diagonal())) == 3:
        return board[0, 2]
    if not np.any(board == 0):
        return 0  # Draw
    return None  # Game ongoing

def draw_board(board):
    plt.figure(figsize=(12, 6))

    # Plot the game board
    plt.subplot(1, 2, 1)
    plt.imshow(np.zeros((GRID_SIZE, GRID_SIZE)), cmap='gray')
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            center = (col, row)
            if board[row, col] == 1:
                plt.plot(center[0], center[1], 'x', color='blue', markersize=40, markeredgewidth=5)
            elif board[row, col] == -1:
                circle = plt.Circle((center[0], center[1]), 0.2, color='red', fill=False, linewidth=5)
                plt.gca().add_artist(circle)
    plt.xticks([])
    plt.yticks([])
    plt.grid(True)

    # Plot the rewards history
    plt.subplot(1, 2, 2)
    plt.plot(rewards_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Rewards over Episodes')

    plt.show()

def build_model():
    model = Sequential([
        Input(shape=(9,)),  # Explicitly use Input layer
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(9, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')
    return model

def train_dqn():
    global EXPLORATION_RATE  # Declare as global
    model = build_model()
    target_model = build_model()
    target_model.set_weights(model.get_weights())
    
    global rewards_history  # Declare as global to be used in draw_board
    rewards_history = []

    for episode in range(EPISODES):
        board = np.zeros((3, 3), dtype=int)
        state = get_state(board)
        total_reward = 0
        done = False
        while not done:
            if random.uniform(0, 1) < EXPLORATION_RATE:
                action = random.choice(get_possible_actions(board))
            else:
                q_values = model.predict(np.array([state]))[0]
                action = np.argmax(q_values)

            row, col = divmod(action, 3)
            board[row, col] = 1
            next_state = get_state(board)
            winner = check_winner(board)
            reward = 1 if winner == 1 else 0 if winner == 0 else -1
            total_reward += reward

            memory.append((state, action, reward, next_state, done))

            if len(memory) > MEMORY_SIZE:
                memory.pop(0)

            if len(memory) >= BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = np.array(states)
                next_states = np.array(next_states)
                targets = model.predict(states)
                next_q_values = target_model.predict(next_states)
                for i in range(BATCH_SIZE):
                    target = rewards[i] + DISCOUNT_FACTOR * np.max(next_q_values[i]) * (1 - dones[i])
                    targets[i][actions[i]] = target
                model.train_on_batch(states, targets)

            if winner is not None:
                done = True

            # Opponent's move
            if not done:
                possible_actions = get_possible_actions(board)
                if possible_actions:
                    opponent_action = random.choice(possible_actions)
                    row, col = divmod(opponent_action, 3)
                    board[row, col] = -1
                    winner = check_winner(board)
                    next_state = get_state(board)
                    reward = -1 if winner == -1 else 0 if winner == 0 else 1
                    total_reward += reward
                    state = next_state

                    memory.append((state, opponent_action, reward, next_state, done))

                    if len(memory) > MEMORY_SIZE:
                        memory.pop(0)

                    if winner is not None:
                        done = True

            state = next_state

        EXPLORATION_RATE = max(MIN_EXPLORATION_RATE, EXPLORATION_RATE * EXPLORATION_DECAY)
        target_model.set_weights(model.get_weights())

        rewards_history.append(total_reward)

        # Render the game at intervals
        if episode % RENDER_INTERVAL == 0:
            clear_output(wait=True)
            draw_board(board)
            plt.title(f'Episode: {episode}')

    return model, rewards_history

def play_game(model):
    board = np.zeros((3, 3), dtype=int)
    draw_board(board)
    while True:
        x, y = map(int, input("Enter row and column (e.g., 1 2): ").split())
        if board[x, y] == 0:
            board[x, y] = 1
            draw_board(board)
            if check_winner(board) is not None:
                break
            state = get_state(board)
            q_values = model.predict(np.array([state]))[0]
            action = np.argmax(q_values)
            row, col = divmod(action, 3)
            if board[row, col] == 0:
                board[row, col] = -1
                draw_board(board)
                if check_winner(board) is not None:
                    break

# Main
if __name__ == "__main__":
    trained_model, rewards_history = train_dqn()
    play_game(trained_model)
