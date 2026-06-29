from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector
from random import Random


class SchellingAgent(Agent):
    """
    Schelling segregation agent
    """

    def __init__(self, pos, model, agent_type, random_move):
        """
        Create a new Schelling agent.

        Args:
           unique_id: Unique identifier for the agent.
           x, y: Agent initial location.
           agent_type: Indicator for the agent's type (minority=1, majority=0)
        """
        super().__init__(pos, model)
        self.pos = pos
        self.type = agent_type
        self.random_move = random_move

    def step(self):
        similar = 0
        for neighbor in self.model.grid.iter_neighbors(self.pos, True):
            if neighbor.type == self.type:
                similar += 1

        # If unhappy, move:
        if similar < self.model.homophily:
            self.model.grid.move_to_empty(self)
        else:
            self.model.happy += 1
            
    @property
    def random(self) -> Random:
        return self.random_move


class Schelling(Model):
    """
    Model class for the Schelling segregation model.
    """

    def __init__(self, width=20, height=20, density=0.8, minority_pc=0.2, homophily=3,
                 seed_init = 0,
                 seed_activate = 0,
                 seed_move = 0,
                 ):
        """ """

        self.width = width
        self.height = height
        self.density = density
        self.minority_pc = minority_pc
        self.homophily = homophily

        self.schedule = RandomActivation(self)
        self.grid = SingleGrid(width, height, torus=True)

        self.happy = 0
        self.Segregated_Agents = 0
        self.datacollector = DataCollector(
            {"happy": "happy",  # Model-level count of happy agents
            # For testing purposes, agent's individual x and y
            "Segregated_Agents": lambda m: m.get_segregation()},
            {"x": lambda a: a.pos[0], "y": lambda a: a.pos[1]},
        )

        # used to shuffle agents during activation
        self.random = Random(seed_activate)
        random_move = Random(seed_move)
        random_init = Random(seed_init)
        
        # Set up agents
        # We use a grid iterator that returns
        # the coordinates of a cell as well as
        # its contents. (coord_iter)
        for cell in self.grid.coord_iter():
            x = cell[1]
            y = cell[2]
            if random_init.random() < self.density:
                if random_init.random() < self.minority_pc:
                    agent_type = 1
                else:
                    agent_type = 0

                agent = SchellingAgent((x, y), self, agent_type, random_move)
                self.grid.position_agent(agent, (x, y))
                self.schedule.add(agent)

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        """
        Run one step of the model. If All agents are happy, halt the model.
        """
        self.happy = 0  # Reset counter of happy agents
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)

        if self.happy == self.schedule.get_agent_count():
            self.running = False

        self.Segregated_Agents = self.get_segregation()
        
    def get_segregation(self):
        """
        Find the % of agents that only have neighbors of their same type.
        """
        segregated_agents = 0
        for agent in self.schedule.agents:
            segregated = True
            for neighbor in self.grid.iter_neighbors(agent.pos, True):
                if neighbor.type != agent.type:
                    segregated = False
                    break
            if segregated:
                segregated_agents += 1
        return segregated_agents / self.schedule.get_agent_count()