import logging
import math
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter

plt.ion()  # Turn on interactive mode

# Constants
G = 9.81  # Gravity (m/s^2)
V_e = 2570  # Exit velocity at nozzle (m/s)
p_inf = 40.34  # Average air pressure (kPa)
p_e = 70.9275  # Exit pressure at nozzle (kPa)
A_e = (0.9144 / 2) ** 2 * math.pi  # Exit area of nozzle (m^2)
m_0 = 28122.7269  # Total initial rocket mass (kg)
FRAME_TIME = 0.1  # Time interval

initialheight = 200 # Initial height of rocket (m)
initialvel = 0 # Initial velocity of rocket (m/s)

# Dynamics class
class Dynamics(nn.Module):
    def __init__(self):
        super(Dynamics, self).__init__()

    def forward(self, state, action):
        # Ensure no in-place operations on 'state'
        new_state = state.clone()

        # Unpack state
        y, vy = new_state[0, 1], new_state[0, 3]
        m_dot_p = action[0, 0]

        # Rocket dynamics
        a_r = (m_dot_p * V_e + (p_e - p_inf) * A_e - (m_0 - m_dot_p * FRAME_TIME) * G) / (m_0 - m_dot_p * FRAME_TIME)
        vy_new = vy + a_r * FRAME_TIME
        y_new = y + vy * FRAME_TIME + 0.5 * a_r * FRAME_TIME ** 2

        # Update state without in-place operations
        new_state[0, 1] = y_new
        new_state[0, 3] = vy_new
        return new_state

# Controller class
class Controller(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output):
        super(Controller, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim_output),
            nn.Sigmoid())

    def forward(self, state):
        action = self.network(state)
        # Scale the output to range 0-1000
        action = action * 1000
        return action

# Simulation class
class Simulation(nn.Module):
    def __init__(self, controller, dynamics, T):
        super(Simulation, self).__init__()
        self.state = self.initialize_state()
        self.controller = controller
        self.dynamics = dynamics
        self.T = T
        self.state_trajectory = []

    def forward(self, state):
        self.action_trajectory = []
        self.state_trajectory = []
        for _ in range(self.T):
            action = self.controller(state)
            state = self.dynamics(state, action)
            self.action_trajectory.append(action)
            self.state_trajectory.append(state.clone())  # Ensure a copy is stored
        return self.error(state)

    @staticmethod
    def initialize_state():
        # Initial state: [x, y, vx, vy, theta]
        state = [[0., initialheight, 0., initialvel, 0.]]  # Starting at height with a downward velocity
        return torch.tensor(state, requires_grad=False).float()

    def error(self, state):
        # Assuming 'state' is the last state in the trajectory
        final_state = self.state_trajectory[-1][0]  # Access the last state tensor
        height_error = (final_state[1])**2  # Penalize final height not being zero
        velocity_error = (final_state[3])**2  # Penalize final velocity not being zero
        return height_error + velocity_error

# Optimize class
class Optimize:
    def __init__(self, simulation):
        self.simulation = simulation
        self.parameters = simulation.controller.parameters()
        self.optimizer = optim.LBFGS(self.parameters, lr=0.01)
        self.loss_list = []
        self.figures = []  # Store figures for each epoch
        self.plot_data = []
        self.epoch_data = []
        self.final_epoch_data = None  # To store the final epoch's data

    def step(self):
        def closure():
            loss = self.simulation(self.simulation.state)
            self.optimizer.zero_grad()
            loss.backward()
            return loss

        self.optimizer.step(closure)
        return closure()

    def train(self, epochs):
        for epoch in range(epochs):
            loss = self.step()
            self.loss_list.append(loss)
            print('[%d] loss: %.3f' % (epoch + 1, loss.item()))
            self.visualize(epoch)  # Store figure for each epoch

        self.display_all_figures()  # Display all figures after training

        if epoch == epochs - 1:  # If it's the final epoch
                self.final_epoch_data = self.epoch_data[-1]  # Store the final epoch's data

    def visualize(self, epoch):
        # Extracting data from the trajectories
        T = self.simulation.T
        data = np.array([self.simulation.state_trajectory[i][0].detach().numpy() for i in range(T)])
        y = data[:, 1]  # Height
        vy = data[:, 3]  # Velocity

        # Calculating acceleration from action_trajectory
        action_data = np.array([self.simulation.action_trajectory[i][0].detach().numpy() for i in range(T)])
        m_dot_p = action_data[:, 0]
        a_r = (m_dot_p * V_e + (p_e - p_inf) * A_e - (m_0 - m_dot_p * FRAME_TIME) * G) / (m_0 - m_dot_p * FRAME_TIME)

        # Time steps
        time_steps = np.arange(0, T * FRAME_TIME, FRAME_TIME)

        # Calculate cumulative fuel consumption
        fuel_spent = np.cumsum(m_dot_p * FRAME_TIME)

        # Store the data for later plotting
        self.epoch_data.append((epoch, time_steps, y, vy, a_r, fuel_spent, m_dot_p))

    def display_all_figures(self):
        # Display plots for all epochs
        figures = []  # List to store figures
        for epoch, time_steps, y, vy, a_r, fuel_spent, m_dot_p in self.epoch_data:
            fig, axs = plt.subplots(5, 1, figsize=(10, 20))  # Adjust for 5 subplots

            # Plotting height
            axs[0].plot(time_steps, y, label='Height (m)')
            axs[0].set_xlabel('Time (s)')
            axs[0].set_ylabel('Height (m)')
            axs[0].set_title(f'Rocket Height Over Time (Epoch {epoch+1})')
            axs[0].grid(True)

            # Plotting velocity
            axs[1].plot(time_steps, vy, label='Velocity (m/s)', color='orange')
            axs[1].set_xlabel('Time (s)')
            axs[1].set_ylabel('Velocity (m/s)')
            axs[1].set_title(f'Rocket Velocity Over Time (Epoch {epoch+1})')
            axs[1].grid(True)

            # Plotting acceleration
            axs[2].plot(time_steps, a_r, label='Acceleration (m/s²)', color='green')
            axs[2].set_xlabel('Time (s)')
            axs[2].set_ylabel('Acceleration (m/s²)')
            axs[2].set_title(f'Rocket Acceleration Over Time (Epoch {epoch+1})')
            axs[2].grid(True)

            # Plotting total fuel spent
            axs[3].plot(time_steps, fuel_spent, label='Total Fuel Spent (kg)', color='red')
            axs[3].set_xlabel('Time (s)')
            axs[3].set_ylabel('Fuel Spent (kg)')
            axs[3].set_title(f'Total Fuel Spent Over Time (Epoch {epoch+1})')
            axs[3].grid(True)

            # Plotting m_dot_p
            axs[4].plot(time_steps, m_dot_p, label='Mass Flow Rate (m_dot_p)', color='black')
            axs[4].set_xlabel('Time (s)')
            axs[4].set_ylabel('Mass Flow Rate (kg/s)')
            axs[4].set_title(f'Mass Flow Rate Over Time (Epoch {epoch+1})')
            axs[4].grid(True)

            plt.tight_layout()
            figures.append(fig)  # Store the figure

        return figures

    def create_animation(self):
        if self.final_epoch_data is None:
            print("No data available for animation.")
            return

        _, time_steps, y, _, _, fuel_spent, m_dot_p = self.final_epoch_data

        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        ax.set_ylim(-25, max(y) + 50)
        rocket_width = 0.5
        rocket_height = 25
        thrust_width = rocket_width - 0.25
        thrust_height = 1
        thrust = plt.Rectangle((4.625, y[0] - thrust_height), thrust_width, thrust_height, color='red')  # Adjusted position and size
        rocket = plt.Rectangle((4.5, y[0]), rocket_width, rocket_height, color='orange')  # Centered and taller
        ax.add_patch(thrust)
        ax.add_patch(rocket)

        # Add a dark grey dashed line to represent the ground
        ax.axhline(y=0, color='darkgrey', linestyle='--')

        def animate(i):
            rocket.set_y(y[i])
            current_thrust_height = m_dot_p[i] / 25 + thrust_height
            thrust.set_y(y[i] - current_thrust_height)  # Adjust y-coordinate based on current thrust height
            thrust.set_height(current_thrust_height)
            return rocket, thrust

        ani = animation.FuncAnimation(fig, animate, frames=len(time_steps), interval=100, blit=True)
        ani.save('E:/ASU/Senior Fall Semester/MAE 598 & 494/Project 1/rocket_landing.gif', writer='pillow')

        # Uncomment the following line if you want to display the animation in the script
        # plt.show()

    
    #def animation(self):
        # Animation code here

# Main execution
T = 100  # Number of time steps
dim_input = 5  # State space dimensions
dim_hidden = 6  # Latent dimensions
dim_output = 1  # Action space dimensions (m_dot_p)

d = Dynamics()
c = Controller(dim_input, dim_hidden, dim_output)
s = Simulation(c, d, T)
o = Optimize(s)
o.train(50)  # Train for 50 epochs
o.create_animation()  # Create and save the animation

# Visualization and animation
plt.show()  # This will keep all figures open