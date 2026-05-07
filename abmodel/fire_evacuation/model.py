import numpy as np
import networkx as nx
import math

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.discrete_space.cell import Coordinate
from mesa.experimental.scenarios import Scenario

from .agent import Human, Wall, FireExit

class FireEvacuationScenario(Scenario):
    """Scenario for Prisoner's Dilemma model.
    
    Attributes
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
    """

    AlARM_BELIEVERS_PROB = 0.9
    TURN_WHEN_BLOCKED_PROB = 0.5
    COOPERATION_MEAN = 0.3
    NERVOUSNESS_MEAN = 0.3
    
    floor_size: int = 14
    human_count: int = 50
    visualise_vision: bool = True
    random_spawn: bool = True
    alarm_believers_prop: float = AlARM_BELIEVERS_PROB
    turnwhenblocked_prop: float = TURN_WHEN_BLOCKED_PROB
    max_speed: int = 1
    cooperation_mean: float = COOPERATION_MEAN
    nervousness_mean: float = NERVOUSNESS_MEAN

    # add parameter for proportion of facilitators here

class FireEvacuation(Model):
    
    MIN_SPEED = 0
    MAX_SPEED = 3
    
    EXTRA_STEPS_PER_IMMOBILE = 10

    COOPERATE_WO_EXIT = False

    
    def __init__(
        self,
        scenario: FireEvacuationScenario = FireEvacuationScenario,
     ):
        """
        Parameters
        ----------
        scenario: FireEvacuationScenario
            applied scenario settings

        Returns
        -------
        None.

        """
        print("Start model...")
        super().__init__(scenario=scenario)
        
        if scenario.human_count > scenario.floor_size ** 2:
            raise ValueError("Number of humans to high for the room!")
 
        self.MAX_SPEED = scenario.max_speed
        self.COOPERATE_WO_EXIT = FireEvacuation.COOPERATE_WO_EXIT
        
        self.stepcounter = -1
        
        # Create floorplan
        floorplan = np.full((scenario.floor_size, scenario.floor_size), 'S')
        floorplan[(0,-1),:]='W'
        floorplan[:,(0,-1)]='W'
        floorplan[math.floor(scenario.floor_size/2),(0,-1)] = 'E'
        floorplan[(0,-1), math.floor(scenario.floor_size/2)] = 'E'

        # Rotate the floorplan so it's interpreted as seen in the text file
        floorplan = np.rot90(floorplan, 3)

        # Init params
        self.width = scenario.floor_size
        self.height = scenario.floor_size
        self.human_count = scenario.human_count
        self.visualise_vision = scenario.visualise_vision

        # Set up grid
        self.grid = OrthogonalMooreGrid((self.width, self.height), torus=False, capacity=1, random=self.random)

        # Used to easily see if a location is a FireExit or Door, since this needs to be done a lot
        self.fire_exits: dict[Coordinate, FireExit] = {}

        # If random spawn is false, spawn_pos_list will contain the list of possible 
        # spawn points according to the floorplan
        self.random_spawn = scenario.random_spawn
        self.spawn_pos_list: list[Coordinate] = []

        self.decisioncount = dict()
        
        # Load floorplan objects
        for (x, y), value in np.ndenumerate(floorplan):
            #pos: Coordinate = (x, y)
            cell = self.grid.find_nearest_cell([x, y])
            
            value = str(value)
            floor_object = None
            if value == "W":
                floor_object = Wall(self)
                floor_object.cell = cell
            elif value == "E":
                floor_object = FireExit(self)
                cell.capacity = None
                floor_object.cell = cell
                self.fire_exits[cell] = floor_object
            elif value == "S":
                self.spawn_pos_list.append(cell)
                

        # Create a graph of traversable routes, used by humans for pathing
        self.graph = nx.Graph()
        for cell in self.grid.all_cells:
            # If the location is empty, or there are no non-traversable objects
            if len(cell.agents) == 0 or not any(not agent.traversable for agent in cell.agents):
                neighbors_cells = cell.get_neighborhood(
                    radius=1, include_center=True,
                )

                for neighbor_cell in neighbors_cells:
                    # If the neighbour position is empty, or no non-traversable 
                    # contents, add an edge
                    if neighbor_cell.is_empty or not any(
                        not agent.traversable
                        for agent in neighbor_cell.agents
                    ):
                        self.graph.add_edge(cell, neighbor_cell)

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
                cell = self.grid.select_random_empty_cell()
            else:  # Place humans at specified spawn locations
                cell = self.rng.choice(self.spawn_pos_list)

            if cell:
                # Create a random human
                speed = self.rng.integers(self.MIN_SPEED + 1, self.MAX_SPEED + 1)

                nervousness = -1
                while nervousness < 0 or nervousness > 1:
                    nervousness = self.rng.normal(loc = scenario.nervousness_mean, scale = 0.2)
                    
                cooperativeness = -1
                while cooperativeness < 0 or cooperativeness > 1:
                    cooperativeness = self.rng.normal(scenario.cooperation_mean)

                belief_distribution = [scenario.alarm_believers_prop, 1 - scenario.alarm_believers_prop]
                believes_alarm = self.rng.choice([True, False], p=belief_distribution)

                orientation = Human.Orientation(self.rng.integers(1,5))
                
                # decide here whether to add a facilitator
                
                human = Human(
                    speed=speed,
                    orientation=orientation,
                    nervousness=nervousness,
                    cooperativeness=cooperativeness,
                    believes_alarm=believes_alarm,
                    turnwhenblocked_prop = scenario.turnwhenblocked_prop,
                    model=self,
                )

                human.cell = cell
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
 
