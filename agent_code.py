import logging
logging.basicConfig(level=logging.DEBUG, filename='agent_log.txt', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
from collections import deque
from environment_code import Action

class ActionSpace:
    def __init__(self, step_sizes):
        """
        Initialize the action space with the step sizes for each operation.
        :param step_sizes: A dictionary specifying the step size for width, height, and position adjustments.
        """
        self.step_sizes = step_sizes
        self.actions = {
            'add_element': self.add_element,
            'remove_element': self.remove_element,
            'increase_width': lambda element, step: self.modify_dimension(element, 'width', step),
            'decrease_width': lambda element, step: self.modify_dimension(element, 'width', -step),
            'increase_height': lambda element, step: self.modify_dimension(element, 'height', step),
            'decrease_height': lambda element, step: self.modify_dimension(element, 'height', -step),
            'move_up': lambda element, step: self.modify_position(element, 'top', -step),
            'move_down': lambda element, step: self.modify_position(element, 'top', step),
            'move_left': lambda element, step: self.modify_position(element, 'left', -step),
            'move_right': lambda element, step: self.modify_position(element, 'left', step)
        }

    def modify_dimension(self, element, dimension, delta):
        """
        Modify the dimension of an element by a specified delta.
        :param element: Dictionary representing the HTML element.
        :param dimension: String, either 'width' or 'height'.
        :param delta: The amount by which to modify the dimension.
        """
        if dimension in element['style']:
            element['style'][dimension] += delta
            element['style'][dimension] = max(0, element['style'][dimension])  # Prevent negative dimensions

    def modify_position(self, element, position, delta):
        """
        Modify the position of an element by a specified delta.
        :param element: Dictionary representing the HTML element.
        :param position: String, either 'top' or 'left'.
        :param delta: The amount by which to modify the position.
        """
        if position in element['style']:
            element['style'][position] += delta

    def add_element(self, element_type, style):
        """
        Add a new element with a specific style.
        :param element_type: Type of the element to add (e.g., 'div', 'button').
        :param style: Dictionary with style attributes (position, size).
        """
        return {'type': element_type, 'style': style}

    def remove_element(self, element_id, elements):
        """
        Remove an element by its ID.
        :param element_id: ID of the element to remove.
        :param elements: List containing all elements.
        """
        elements = [el for el in elements if el['id'] != element_id]
        return elements

    def execute_action(self, action_type, element=None, step=None, element_type=None, style=None):
        """
        Execute an action based on the action type.
        :param action_type: Type of action to execute.
        :param element: The element on which the action is to be executed (if applicable).
        :param step: The step size for the action (if applicable).
        :param element_type: The type of a new element to add (if applicable).
        :param style: The style of the new element (if applicable).
        """
        if action_type in ['increase_width', 'decrease_width', 'increase_height', 'decrease_height',
                           'move_up', 'move_down', 'move_left', 'move_right']:
            self.actions[action_type](element, self.step_sizes[action_type.split('_')[1]])
        elif action_type == 'add_element':
            return self.actions[action_type](element_type, style)
        elif action_type == 'remove_element':
            return self.actions[action_type](element, style)
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        """Neural Net for Deep-Q learning Model."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),  # Define input shape explicitly here
            tf.keras.layers.Dense(48, activation='relu'),
            tf.keras.layers.Dropout(0.25),  # Dropout layer for regularization
            tf.keras.layers.Dense(48, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """Copy weights from model to target_model."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # Random exploration: Choose a random action
            action_index = random.randrange(self.action_size)
        else:
            # Exploitation: Choose the best action based on the model's prediction
            act_values = self.model.predict(state)
            action_index = np.argmax(act_values[0])
        
        # Convert action index to Action object
        return self.index_to_action(action_index)

    def index_to_action(self, index):
        # Maps indices to Action instances
        action_types = ['add', 'modify', 'remove', 'rearrange']
        # Example mapping, adjust according to your setup
        action_type = action_types[index % len(action_types)]
        # You may need to include more logic here to determine other properties of Action
        return Action(action_type=action_type)

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.target_model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name + 'h5')
        with open(filename + '_params.json', 'r') as f:
            params = json.load(f)
            self.epsilon = params['epsilon']
        logging.info(f"Model loaded from {filename}.h5 and params from {filename}_params.json")

    def save(self, name):
        self.model.save_weights(name + 'h5')
        with open(filename + '_params.json', 'w') as f:
            json.dump({'epsilon': self.epsilon}, f)
        logging.info(f"Model saved to {filename}.h5 and params to {filename}_params.json")
