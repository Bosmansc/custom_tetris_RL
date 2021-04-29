import random


from time import sleep 
from engine import TetrisEngine

class BasicAgent():
    def __init__(self):
        # Initializes a Tetris playing field of width 10 and height 20.
        self.env = TetrisEngine()

    def run(self):
        # Loop to keep playing games
        while True:
            # Variable to indicate whether the game has ended or not
            done = False
            # Resets the environment
            state = self.env.reset()
            
            # Loop that keeps making moves as long as the game hasn't ended yet
            while not done:
                # Picks a random action
                action = random.randint(0, 5)
                action = 5
                # Performs the action in the game engine
                next_state, reward, done, info = self.env.step(action)
                # Render the game state
                self.env.render()
                # Sleep to make sure a human can follow the gameplay
                sleep(0.05)
                    
                    
if __name__ == '__main__':
    agent = BasicAgent()
    agent.run()
