import numpy as np
import networkx as nx
import math

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import Coordinate, MultiGrid

from .agent import Human, Wall, FireExit


class FireEvacuation(Model):
    
    MIN_SPEED = 0
    MAX_SPEED = 3
    
    EXTRA_STEPS_PER_IMMOBILE = 10

    COOPERATE_WO_EXIT = False
    AlARM_BELIEVERS_PROB = 0.9
    TURN_WHEN_BLOCKED_PROB = 0.5
    COOPERATION_MEAN = 0.3
    NERVOUSNESS_MEAN = 0.3
    
    def __init__(
        self,
        floor_size: int = 14,
        human_count: int = 50,
        visualise_vision = True,
        random_spawn = True,
        alarm_believers_prop = AlARM_BELIEVERS_PROB,
        turnwhenblocked_prop = TURN_WHEN_BLOCKED_PROB,
        max_speed = 1,
        cooperation_mean = COOPERATION_MEAN,
        nervousness_mean = NERVOUSNESS_MEAN,
        seed = 1,
        
        # add parameter for proportion of facilitators here
     ):
        """
        

        Parameters
        ----------
        floor_size : int
            Size of the room excluding walls.
        human_count : int
            Number Of Human Agents.
        visualise_vision : bool
            When true, show agents' vision on grid.
        random_spawn : bool
            Random spawn of initial positions.
        alarm_believers_prop: float
            Proportion of Alarm Believers
        turnwhenblocked_prop: float
            Probabilty to turn for an agent who is blockec
        max_speed: int
            maximum agent speed in cells per step
        cooperation_mean : float
            Mean Cooperation.
        nervousness_mean : float
            Mean nervousness
        seed : int
            Random seed for all random processes.

        Returns
        -------
        None.

        """
        super().__init__(seed=seed)
        self.rng = np.random.default_rng(seed)
        
        if human_count > floor_size ** 2:
            raise ValueError("Number of humans to high for the room!")
 
        self.MAX_SPEED = max_speed
        self.COOPERATE_WO_EXIT = FireEvacuation.COOPERATE_WO_EXIT
        
        self.stepcounter = -1
        
        # Create floorplan
        floorplan = np.full((floor_size, floor_size), '_')
        floorplan[(0,-1),:]='W'
        floorplan[:,(0,-1)]='W'
        floorplan[math.floor(floor_size/2),(0,-1)] = 'E'
        floorplan[(0,-1), math.floor(floor_size/2)] = 'E'

        # Rotate the floorplan so it's interpreted as seen in the text file
        floorplan = np.rot90(floorplan, 3)

        # Init params
        self.width = floor_size
        self.height = floor_size
        self.human_count = human_count
        self.visualise_vision = visualise_vision

        # Set up grid
        self.grid = MultiGrid(floor_size, floor_size, torus=False)

        # Used to easily see if a location is a FireExit or Door, since this needs to be done a lot
        self.fire_exits: dict[Coordinate, FireExit] = {}

        # If random spawn is false, spawn_pos_list will contain the list of possible 
        # spawn points according to the floorplan
        self.random_spawn = random_spawn
        self.spawn_pos_list: list[Coordinate] = []

        self.decisioncount = dict()
        
        # Load floorplan objects
        for (x, y), value in np.ndenumerate(floorplan):
            pos: Coordinate = (x, y)

            value = str(value)
            floor_object = None
            if value == "W":
                floor_object = Wall(self)
            elif value == "E":
                floor_object = FireExit(self)
                self.fire_exits[pos] = floor_object
            elif value == "S":
                self.spawn_pos_list.append(pos)
            if floor_object:
                self.grid.place_agent(floor_object, pos)

        # Create a graph of traversable routes, used by humans for pathing
        self.graph = nx.Graph()
        for agents, pos in self.grid.coord_iter():
            # If the location is empty, or there are no non-traversable objects
            if len(agents) == 0 or not any(not agent.traversable for agent in agents):
                neighbors_pos = self.grid.get_neighborhood(
                    pos, moore=True, include_center=True, radius=1
                )

                for neighbor_pos in neighbors_pos:
                    # If the neighbour position is empty, or no non-traversable 
                    # contents, add an edge
                    if self.grid.is_cell_empty(neighbor_pos) or not any(
                        not agent.traversable
                        for agent in self.grid.get_cell_list_contents(neighbor_pos)
                    ):
                        self.graph.add_edge(pos, neighbor_pos)

        # Collects statistics from our model run
        self.datacollector = DataCollector(
            {
                "NumEscaped" : lambda m: self.get_num_escaped(m),
                "AvgNervousness": lambda m: self.get_human_nervousness(m),
                "AvgSpeed": lambda m: self.get_human_speed(m),
             }
        )
        
        # Start placing humans
        for _i in range(0, self.human_count):
            if self.random_spawn:  # Place humans randomly
                pos = tuple(self.rng.choice(tuple(self.grid.empties)))
            else:  # Place humans at specified spawn locations
                pos = self.rng.choice(self.spawn_pos_list)

            if pos:
                # Create a random human
                speed = self.rng.integers(self.MIN_SPEED, self.MAX_SPEED + 1)

                nervousness = -1
                while nervousness < 0 or nervousness > 1:
                    nervousness = self.rng.normal(loc = nervousness_mean, scale = 0.2)
                    
                cooperativeness = -1
                while cooperativeness < 0 or cooperativeness > 1:
                    cooperativeness = self.rng.normal(cooperation_mean)

                belief_distribution = [alarm_believers_prop, 1 - alarm_believers_prop]
                believes_alarm = self.rng.choice([True, False], p=belief_distribution)

                orientation = Human.Orientation(self.rng.integers(1,5))
                
                # decide here whether to add a facilitator
                
                human = Human(
                    speed=speed,
                    orientation=orientation,
                    nervousness=nervousness,
                    cooperativeness=cooperativeness,
                    believes_alarm=believes_alarm,
                    turnwhenblocked_prop = turnwhenblocked_prop,
                    model=self,
                )

                self.grid.place_agent(human, pos)
            else:
                print("No tile empty for human placement!")

        self.running = True

    def step(self):
        """
        Advance the model by one step.
        """

        self.agents.shuffle_do("step")
        self.datacollector.collect(self)

        # If all humans escaped, stop the model and collect the results
        if self.get_num_escaped(self) == self.human_count:
            self.running = False
        
        # In case there are Humans unable to move let the model run extra steps
        # to indicate the insufficient escape: 
        if self.stepcounter == 0:
            self.running = False
        elif self.stepcounter > 0:
            self.stepcounter -=1
        elif self.get_human_speed(self) == 0:
            self.stepcounter = FireEvacuation.EXTRA_STEPS_PER_IMMOBILE *\
                sum(map(lambda agent : isinstance(agent, Human) 
                            and not agent.escaped, self.agents))
                
    def run(self, n):
        """Run the model for n steps."""
        for _ in range(n):
            self.step()

     
    @staticmethod     
    def get_human_nervousness(model):
        count = 0
        nervousness = 0
        for agent in model.agents:
            if isinstance(agent, Human) and not agent.escaped:
                nervousness += agent.nervousness
                count +=1
        if count == 0:
            return 0
        return nervousness/count
 
    
    @staticmethod     
    def get_human_speed(model):
        count = 0
        speed = 0
        for agent in model.agents:
            if isinstance(agent, Human) and not agent.escaped:
                speed += agent.speed
                count +=1
        if count == 0:
            return 0
        return speed/count


    @staticmethod
    def get_num_escaped(model):
        """
        Helper method to count escaped humans
        """
        count = 0
        for agent in model.agents:
            if isinstance(agent, Human) and agent.escaped == True:
                count += 1

        return count
 
