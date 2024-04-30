import os
import numpy as np
import json
import environment_code
import agent_code

html_templates = {
    "button": '<button {attributes} style="{style}">Button</button>',
    "checkbox": '<label {attributes} style="{style}"><input type="checkbox"> Checkbox</label>',
    "container": '<div {attributes} style="{style}">Container Content</div>',
    "icon-button": '<button {attributes} style="{style}"><img src="icon.png" alt="icon" style="width: 20px; height: 20px;"> Icon Button</button>',
    "image": '<img  {attributes} src="image.jpg" alt="Description" style="{style}">',
    "input": '<input {attributes} type="text" placeholder="Enter text" style="{style}">',
    "label": '<label {attributes} for="inputExample" style="{style}">Label:</label>',
    "link": '<a href="http://example.com" style="{style}">Visit Example</a>',
    "number-input": '<input type="number" placeholder="Enter number" style="{style}">',
    "radio": '<label style="{style}"><input type="radio" name="radioExample"> Radio Button</label>',
    "search": '<input type="search" placeholder="Search here" style="{style}">',
    "select": '<select style="{style}"><option value="option1">Option 1</option><option value="option2">Option 2</option></select>',
    "slider": '<input type="range" min="1" max="100" value="50" style="{style}">',
    "table": '<table style="{style}"><tr><th>Header 1</th><th>Header 2</th></tr><tr><td>Data 1</td><td>Data 2</td></tr></table>',
    "text": '<span style="{style}">Some text here</span>',
    "textarea": '<textarea placeholder="Enter multi-line text" style="{style}"></textarea>',
    "textbox": '<input type="text" placeholder="Enter text" style="{style}">',
    "toggle": '<label style="{style}"><input type="checkbox"> Toggle</label>',
    "pagination": '<div style="{style}"><a href="#">&laquo;</a> <a href="#">1</a> <a href="#">2</a> <a href="#">3</a> <a href="#">&raquo;</a></div>',
    "paragraph": '<p style="{style}">A paragraph of text.</p>',
    "carousel": '<div style="{style}"><img src="slide1.jpg" alt="Slide 1"> <img src="slide2.jpg" alt="Slide 2"> <img src="slide3.jpg" alt="Slide 3"></div>',
    "heading": '<h1 style="{style}">Heading Text</h1>'
}

def load_json_data(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

def train_agent(env, agent, num_episodes=1000, max_steps_per_episode=100, batch_size=32, save_interval=100):
    for e in range(num_episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.state_size])
        total_reward = 0

        for step in range(max_steps_per_episode):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, env.state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        if e % save_interval == 0:
            agent.save(f'model_checkpoint_{e}')

        print(f"Episode: {e+1}, Reward: {total_reward}, Epsilon: {agent.epsilon}")

def train_on_sketches(json_data, sketches_dir, num_episodes=1000):
    # Iterate over each item in the JSON data
    for sketch_filename, sketch_info in json_data.items():
        # Construct the full path to the sketch image
        sketch_path = os.path.join(sketches_dir, sketch_filename)

        # Check if the file exists and is an image
        if os.path.isfile(sketch_path) and sketch_filename.endswith('.jpg'):  # You can add more conditions for other image formats
            bboxes = sketch_info['bboxes']
            labels = sketch_info['labels']
            
            # Initialize the environment with the current sketch data
            env = environment_code.HTMLDesignerEnv(html_templates=html_templates, sketch_path=sketch_path, bboxes=bboxes, labels=labels)
            
            # Initialize the agent
            agent = agent_code.DQNAgent(state_size=100, action_size=5)  # Adjust according to your specific sizes
            
            # Train the agent using the initialized environment and agent
            train_agent(env, agent, num_episodes)
        else:
            print(f"Skipped: {sketch_filename} does not exist in the specified directory.")

