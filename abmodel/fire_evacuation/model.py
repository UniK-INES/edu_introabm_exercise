import numpy as np
import networkx as nx
import math
import logging
import pandas as pd

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.discrete_space import OrthogonalMooreGrid, OrthogonalVonNeumannGrid
from mesa.discrete_space.cell import Coordinate
from mesa.discrete_space import CellCollection, Network, Cell
from mesa.experimental.scenarios import Scenario

from .agent import Human, Wall, FireExit, Facilitator

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
       maxsight: int
            maximum patches an agent can see
        distancenoise: boolean
            if true noise is added to distance perception
        distancenoiselevel: float
            level of noise in perceiving distances
        interact_neumann: float
            probability to interact via von-neumann neighbours            
        interact_moore: float
            probability to interact via moore neighbours
        interact_swnetwork:
            probability to interact via network neighbours
        select_initiator: boolean
            select initiator
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
    facilitators_percentage:float = 20
    max_steps: int = 200
    panic_rush = True
    agentmemorysize = 5
    predictcrowd = False
    maxsight = math.inf
    distancenoise = False
    distancenoisefactor = 1.0
    interact_neumann = None
    interact_moore = None
    interact_swnetwork = None
    select_initiator = False

logger = logging.getLogger("FireEvacuation")

class FireEvacuation(Model):
    
    COUNTER_TURN = "TURN"
    
    MIN_SPEED = 0
    MAX_SPEED = 3
    
    EXTRA_STEPS_PER_IMMOBILE = 10

    COOPERATE_WO_EXIT = False
    
    def __init__(
        self,
        scenario: FireEvacuationScenario = FireEvacuationScenario,
        agentmemories: pd.DataFrame = None,
        rngl = None,
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
        super().__init__(scenario=scenario)
        
        if scenario.human_count > (scenario.floor_size-2) ** 2:
            raise ValueError("Number of humans to high for the room!")
 
        self.MAX_SPEED = scenario.max_speed
        self.COOPERATE_WO_EXIT = FireEvacuation.COOPERATE_WO_EXIT
                
        self.switches = {
            'PREDICT_CROWD': scenario.predictcrowd,
            'DISTANCE_NOISE': scenario.distancenoise,
        }
                
        self.stepcounter = -1
        self.agentmemories = agentmemories
        self.rngl = rngl
        self.random = scenario.rng
        
        if not agentmemories is None:
            self.modelrun = np.max(agentmemories['rep'])
        else:
            self.modelrun = -1

        # Create floorplan
        floorplan = np.full((scenario.floor_size, scenario.floor_size), '_')
        floorplan[(0,-1),:]='W'
        floorplan[:,(0,-1)]='W'
        floorplan[math.floor(scenario.floor_size/2),(0,-1)] = 'E'
        floorplan[(0,-1), math.floor(scenario.floor_size/2)] = 'E'
        
        # distribute agent positions at the south:
        for i in range(scenario.human_count):
            floorplan[1+(i % (scenario.floor_size - 2)), 1 + math.floor(i / (scenario.floor_size - 2))] = 'S'

        # Rotate the floorplan so it's interpreted as seen in the text file
        floorplan = np.rot90(floorplan, 3)

        # Init params
        self.width = scenario.floor_size
        self.height = scenario.floor_size
        self.human_count = scenario.human_count
        self.visualise_vision = scenario.visualise_vision

        # Set up grid
        if scenario.interact_neumann is not None:
            self.grid = OrthogonalVonNeumannGrid((self.width, self.height), torus=False, capacity=1, random=self.rng)
        else:
            self.grid = OrthogonalMooreGrid((self.width, self.height), torus=False, capacity=1, random=self.rng)

        # Used to easily see if a location is a FireExit, since this needs to be done a lot
        self.fire_exits: dict[Coordinate, FireExit] = {}

        # If random spawn is false, spawn_pos_list will contain the list of possible 
        # spawn points according to the floorplan
        self.random_spawn = scenario.random_spawn
        self.spawn_pos_list: list[Coordinate] = []

        self.decisioncount = dict()
        self.exitscount = dict()
        
        if not (scenario.interact_neumann is None and 
                scenario.interact_moore is None and
                scenario.interact_swnetwork is None):
            interactionmatrix = {"neumann": scenario.interact_neumann, 
                                 "moore": scenario.interact_moore,
                                 "swnetwork": scenario.interact_swnetwork}
        else:
            interactionmatrix = None 
            
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
                "NumHelping" : lambda m: self.get_num_helping(m),
                "TurnCount": lambda m: self.get_decision_count(self.COUNTER_TURN),
                
                "UpdateSpeedCount": lambda m: self.get_decision_count(Human.DECISION_SPEED),
                "CooperateCount": lambda m: self.get_decision_count(Human.DECISION_COOPERATE),
                "PlanTargetCount": lambda m: self.get_decision_count(Human.DECISION_PLAN_TARGET),
                "RandomWalkCount": lambda m: self.get_decision_count(Human.DECISION_RANDOM_WALK),
                
                "EscapedWest": lambda m: self.get_escaped_exit(list(self.fire_exits)[0]),
                "EscapedSouth": lambda m: self.get_escaped_exit(list(self.fire_exits)[1]),
                "EscapedNorth": lambda m: self.get_escaped_exit(list(self.fire_exits)[2]),
                "EscapedEast": lambda m: self.get_escaped_exit(list(self.fire_exits)[3]),
             }
        )
        
        ##################################
        # Network Initialisation
        ##################################
        
        self.G = nx.watts_strogatz_graph(n=self.human_count, k=5, p=0.3, seed = self.random)
        self.net = Network(self.G, capacity=1, random=self.random)
        nodes = enumerate(self.G.nodes())
                         
        ################################## 
        # Agent creation
        ##################################
        
        # Start placing humans
        for i in range(0, self.human_count):
            if self.random_spawn:  # Place humans randomly
                cell = self.grid.select_random_empty_cell()
            else:  # Place humans at specified spawn locations
                cell = self.rng.choice(self.spawn_pos_list)
                self.spawn_pos_list.remove(cell)

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
                if interactionmatrix is None:
                    believes_alarm = self.rng.choice([True, False], p=belief_distribution)
                else:
                    believes_alarm = False

                orientation = Human.Orientation(self.rng.integers(1,5))
                
                # decide here whether to add a facilitator
                if i < (scenario.human_count*scenario.facilitators_percentage) // 100.0:
                    human = Facilitator(
                        speed=speed,
                        orientation = orientation,
                        nervousness = nervousness,
                        cooperativeness=cooperativeness,
                        switches = self.switches,
                        memories = self.agentmemories,
                        memorysize = scenario.agentmemorysize,
                        turnwhenblocked_prop = scenario.turnwhenblocked_prop,
                        maxsight = scenario.maxsight,
                        interactionmatrix = interactionmatrix,
                        model=self,
                    )
                else:
                
                    human = Human(
                        speed=speed,
                        orientation=orientation,
                        nervousness=nervousness,
                        cooperativeness=cooperativeness,
                        switches = self.switches,
                        memories = self.agentmemories,
                        memorysize = scenario.agentmemorysize,
                        maxsight = scenario.maxsight,
                        interactionmatrix = interactionmatrix,
                        believes_alarm=believes_alarm,
                        turnwhenblocked_prop = scenario.turnwhenblocked_prop,
                        model=self,
                    )

                human.cell = cell
                
                # add to network
                _ , node = next(nodes)
                human.node = self.net._cells[node]
                self.net._cells[node].add_agent(human)
                logger.debug(f"Initialised agent {human} at {human.cell}")
            else:
                print("No tile empty for human placement!")

        # select random agent to propagate alarm
        if interactionmatrix is not None:
            if scenario.select_initiator:
                # implement initiator selection here
                initiator = self.rng.choice(self.agents)
            else:
                initiator = self.rng.choice(self.agents)
                
            initiator.believes_alarm = True
        
        self.running = True
        logger.info("Model initialised")


    def create_memories(self):
    # create agent memory
        for agent in self.agents:
            if isinstance(agent, Human):
                data = pd.DataFrame({'rep':self.modelrun + 1, 'agent':agent.unique_id, 
                        'cooperativeness':agent.cooperativeness, 
                        #'numsteps2escape': self.schedule.steps},index = [0])
                        'numsteps2escape':agent.numsteps2escape}, index=[0])
                if not self.agentmemories is None:
                    self.agentmemories = pd.concat([self.agentmemories, data])
                else:
                    self.agentmemories = data


    def step(self):
        """
        Advance the model by one step.
        """

        logger.info("Running step " + str(self.time))
        self.agents.shuffle_do("step")
        
        self.datacollector.collect(self)

        # If all humans escaped, stop the model and collect the results
        if self.get_num_escaped(self) == self.human_count:
            self.running = False
            self.create_memories()

        if self.time >= self.scenario.max_steps:
            self.running = False
            self.create_memories()
            
        # In case there are Humans unable to move let the model run extra steps
        # to indicate the insufficient escape: 
        if self.stepcounter == 0:
            self.running = False
            self.create_memories()
                
        if self.stepcounter >= 0:
            self.stepcounter -=1
        elif self.get_human_speed(self) == 0:
            self.stepcounter = FireEvacuation.EXTRA_STEPS_PER_IMMOBILE *\
                sum(map(lambda agent : isinstance(agent, Human) 
                            and not agent.escaped, self.agents))

    def get_agentmemories(self):
        return self.agentmemories
     
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

    def escaped(self, cell):
        if cell not in self.exitscount:
            self.exitscount[cell] = 0
        self.exitscount[cell] +=1 

    @staticmethod
    def get_num_escaped(model):
        """
        Helper method to count escaped humans
        """      
        return len(model.agents.select(
            lambda a: isinstance(a, Human) and a.escaped))

    def get_escaped_exit(self, pos):
        if pos not in self.exitscount.keys():
            return 0
        else:
            return self.exitscount[pos]
        
    @staticmethod
    def get_num_helping(model):
        """
        Helper method to count currently helping humans
        """      
        return len(model.agents.select(lambda a: isinstance(a, Human) and a.humantohelp is not None))

    def increment_decision_count(self, decision):
        """
        Increments the decision counter identified by decision by one.
        Used to count decision all agents do during a step or simulation run.
        
        Args:
            decision: identifier for the specific kind of decision (eg. "TURN")
        """
        if decision not in self.decisioncount:
            self.decisioncount[decision] = 0
        self.decisioncount[decision] +=1 
    

    def get_decision_count(self, decision):
        """
        Retrieve the number of performed decisions (counted when calling
        increment_decision_count(decision)) of the specified kind (decision).
        
        Args:
            decision: identifier for the specific kind of decision (eg. "TURN")
        """
        if decision not in self.decisioncount:
            return 0
        return self.decisioncount[decision]
