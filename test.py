import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

# Sudoku oyunu için gerekli fonksiyonlar
def create_board():
    # Sudoku tahtasını oluştur
    board = np.zeros((9, 9), dtype=int)
    return board

def is_valid_move(board, row, col, num):
    # Belirli bir hücreye num değerini yerleştirip yerleştiremeyeceğini kontrol et
    # Kurallar: Aynı satırda, aynı sütunda ve aynı 3x3 bölgede aynı sayı olmamalıdır
    if num in board[row, :]:
        return False
    if num in board[:, col]:
        return False
    if num in board[row//3*3:(row//3+1)*3, col//3*3:(col//3+1)*3]:
        return False
    return True

def is_board_full(board):
    # Sudoku tahtasının dolu olup olmadığını kontrol et
    return np.all(board != 0)

def print_board(board):
    # Sudoku tahtasını ekrana yazdır
    for row in range(9):
        for col in range(9):
            print(board[row, col], end=' ')
        print()

# Sudoku oyunu için gerekli değişkenler
state_size = 81  # 9x9 Sudoku tahtası
action_size = 9  # 1-9 arası sayılar

# DQNAgent oluştur
agent = DQNAgent(state_size, action_size)

# Oyun döngüsü
episodes = 1000
batch_size = 32
for episode in range(episodes):
    # Sudoku tahtasını oluştur
    board = create_board()
    done = False
    while not done:
        # Eğer tahta dolu ise oyunu bitir
        if is_board_full(board):
            done = True
            break

        # Tahtayı DQNAgent'a uygun hale getir
        state = board.flatten().reshape(1, -1)

        # DQNAgent'tan bir hamle iste
        action = agent.act(state)

        # Hamleyi tahtaya uygula
        row = action // 9
        col = action % 9
        num = board[row, col]
        if num == 0:
            board[row, col] = random.choice(range(1, 10))

        # Hamlenin geçerli olup olmadığını kontrol et
        valid_move = is_valid_move(board, row, col, num)
        if not valid_move:
            board[row, col] = 0

        # Oyunu bitirme koşulunu kontrol et
        done = is_board_full(board)

        # Sonraki durumu al
        next_state = board.flatten().reshape(1, -1)

        # Ödülü hesapla
        reward = 1 if done else 0

        # DQNAgent'a deneyimi hatırlat
        agent.remember(state, action, reward, next_state, done)

        # DQNAgent'tan öğren
        agent.replay(batch_size)

    # Her 10 bölümde bir durumu ekrana yazdır
    if episode % 10 == 0:
        print(f"Episode: {episode}")

# Eğitilmiş modeli kaydet
agent.save("sudoku_model.h5")