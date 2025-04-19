import torch
import random
import numpy as np
from Models.ReplayBuffer import ReplayBuffer


class Agent:
    """
    Interacts with and learns from the environment.
    This class is responsible for the agent's learning process.
    """

    def __init__(
        self,
        qnet_local: torch.nn.Module,
        qnet_target: torch.nn.Module,
        buffer_size: int = 100000,
        batch_size: int = 64,
        gamma: float = 0.99,
        tau: float = 1e-2,
        lr: float = 1e-2,
        update_every: int = 64,
        device: str = "cuda",
    ):
        """
        Initialize the Agent.

        :param qnet_local (torch.nn.Module): Local Q-Network
        :param qnet_target (torch.nn.Module): Target Q-Network
        :param device: Device to be used for computations (CPU or GPU)
        """

        self.device: str = device  # GPU or CPU
        self.BUFFER_SIZE: int = buffer_size  # ReplayBuffer size
        self.BATCH_SIZE: int = batch_size  # Minibatch size
        self.GAMMA: float = gamma  # Discount factor
        self.TAU: float = tau  # Soft update of target parameters
        self.LR: float = lr  # Learning rate
        self.UPDATE_EVERY: int = update_every  # How often to update the network
        self.qnet_local: torch.nn.Module = qnet_local.to(self.device)
        self.qnet_target: torch.nn.Module = qnet_target.to(self.device)
        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(
            self.qnet_local.parameters(), lr=self.LR
        )
        self.memory: ReplayBuffer = ReplayBuffer(
            action_size=4,
            buffer_size=self.BUFFER_SIZE,
            batch_size=self.BATCH_SIZE,
            device=self.device,
        )
        self.t_step = 0

    def step(self, state, action, reward, next_step, done):
        """
        Save experience in replay memory, and use random sample from buffer to learn.

        :param state: Current state of the environment
        :param action: Action taken by the agent
        :param reward: Reward received after taking the action
        :param next_step: Next state of the environment after taking the action
        :param done: Boolean indicating if the episode has ended
        """

        # Adiciona a la memoria los datos obtenidos en el paso de movimiento.
        self.memory.add(state, action, reward, next_step, done)
        # Calcula cada cierto tiempo que se debe recalcular los pesos de la red neuronal.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > self.BATCH_SIZE:
                # Se extren algunas muestras de la experiencia previa. Pero aleatorias.
                experience = self.memory.sample()
                # Se le indica a la red que debe aprender. Esto es, actualizar los parámetros de acuerdo con un gradiente descendente.
                self.learn(experience, self.GAMMA)

    def act(self, state, eps=0.0) -> int:
        """
        Returns action for given state as per current policy.

        :param state: Current state of the environment
        :param eps: Epsilon value for epsilon-greedy action selection
        :return: Action to be taken by the agent
        """

        state = torch.tensor(state).float().unsqueeze(0).unsqueeze(0).to(self.device)
        # La red no va a entrenar, sino a evaluar.
        self.qnet_local.eval()
        with torch.no_grad():
            action_values = self.qnet_local(state)
        # La red dejó de evaluar. Ahora puede ser entrenada de nuevo.
        self.qnet_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(4))

    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.

        :param experiences: Batch of experience tuples
        :param gamma: Discount factor
        """

        states, actions, rewards, next_states, dones = experiences
        states2 = torch.cat(
            [torch.tensor(st).float().unsqueeze(0).unsqueeze(0) for st in states], dim=0
        ).to(self.device)
        next_states2 = torch.cat(
            [torch.tensor(st).float().unsqueeze(0).unsqueeze(0) for st in next_states],
            dim=0,
        ).to(self.device)
        criterion = torch.nn.MSELoss()  # Norma L2 para el cálculo del error.
        self.qnet_local.train()  # La red local se va a entrenar.
        self.qnet_target.eval()  # Esta red no se va a modificar, solamente a evaluar lo que produce a la salida.
        predicted_targets = self.qnet_local(states2).gather(
            1, actions
        )  # Esto es lo que evalua la red neuronal local. Lo predicho de las acciones.

        with torch.no_grad():
            labels_next = self.qnet_target(next_states2).detach().max(1)[0].unsqueeze(1)
        # La red target se evalua en el estado siguiente, en Q(s',a). No en el estado actual y se recibe el valor máximo de acción a tomar allí.
        # .detach() ->  Returns a new Tensor, detached from the current graph.
        labels = rewards + (
            gamma * labels_next * (1 - dones)
        )  # Acá se actualizan los valores predichos de los estados que son terminales y de los que no son terminales. ****
        # Se hace la función de error,
        loss = criterion(predicted_targets, labels).to(self.device)
        self.optimizer.zero_grad()  # Reset del gradiente
        loss.backward()  # Gradiente descendente
        self.optimizer.step()  # Actualización de los parámetros.

        # ------------------- update target network ------------------- #
        # Esta actualización puede hacerse cada cierto tiempo completamente copiando los parámetros de una red a otra o de manera suave, haciendo un proceso de suavización exponencial. Pero la red
        # target no puede actualizarse al mismo ritmo que la red locakl.
        self.soft_update(self.qnet_local, self.qnet_target, self.TAU)

    def soft_update(
        self, local_model: torch.nn.Module, target_model: torch.nn.Module, tau: float
    ):
        """
        Soft update model parameters.

        :param local_model: Local model to be updated
        :param target_model: Target model to be updated
        :param tau: Soft update parameter
        """

        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1 - tau) * target_param.data
            )
