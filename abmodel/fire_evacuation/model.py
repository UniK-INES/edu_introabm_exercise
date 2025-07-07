import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import math
import logging
import pandas as pd

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import Coordinate, MultiGrid, NetworkGrid

from .agent import Human, Facilitator, Wall, FireExit

logger = logging.getLogger("FireEvacuation")

class FireEvacuation(Model):
    
    COUNTER_TURN = "TURN"
    
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
        floor_size: int = 12,
        human_count: int = 100,
        visualise_vision = True,
        random_spawn = True,
        alarm_believers_prop = AlARM_BELIEVERS_PROB,
        turnwhenblocked_prop = TURN_WHEN_BLOCKED_PROB,
        max_speed = 1,
        cooperation_mean = COOPERATION_MEAN,
        nervousness_mean = NERVOUSNESS_MEAN,
        predictcrowd = False,
        agentmemories: pd.DataFrame = None,
        agentmemorysize = 5,
        maxsight = math.inf,
        distancenoise = False,
        distancenoisefactor = 1.0,
        interact_neumann = None,
        interact_moore = None,
        interact_swnetwork = None,
        select_initiator = False,
        rngl = None,
        seed = 1,
        facilitators_percentage = 20
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
        predictcrowd: boolean
            if true agents attempt to predict crowds
        agentmemories: pd.DataFrame
            agent memories
        agentmemorysize: int
            number osf stored entries in memory
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
        seed : int
            Random seed for all random processes.
        facilitator_percentage: float
            Percentage of facilitators to initialise (0..100)

        Returns
        -------
        None.

        """
        super().__init__(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.rngl = rngl
        
        if human_count > floor_size ** 2:
            raise ValueError("Number of humans to high for the room!")
 
        self.MAX_SPEED = max_speed
        self.COOPERATE_WO_EXIT = FireEvacuation.COOPERATE_WO_EXIT
        
        self.switches = {
            'PREDICT_CROWD': predictcrowd,
            'DISTANCE_NOISE': distancenoise,
        }
                
        self.stepcounter = -1
        self.agentmemories = agentmemories
        
        if not agentmemories is None:
            self.modelrun = np.max(agentmemories['rep'])
        else:
            self.modelrun = -1

        # Create floorplan
        floorplan = np.full((floor_size + 2, floor_size + 2), '_')
        floorplan[(0,-1),:]='W'
        floorplan[:,(0,-1)]='W'
        floorplan[math.floor((floor_size + 2)/2),(0,-1)] = 'E'
        floorplan[(0,-1), math.floor((floor_size + 2)/2)] = 'E'
        
        # Create floorplan with thicker walls
        floorplan = np.full((floor_size + 4, floor_size + 4), '_')
        floorplan[(0,1,-2,-1),:]='W'
        floorplan[:,(0,1,-2,-1)]='W'
        floorplan[math.floor((floor_size + 4)/2),(0,-1)] = 'E'
        floorplan[(0,-1), math.floor((floor_size + 4)/2)] = 'E'
        
        floorplan[math.floor((floor_size + 4)/2),(1,-2)] = None
        floorplan[(1,-2), math.floor((floor_size + 4)/2)] = None
        
        # distribute agent positions at the south:
        for i in range(human_count):
            floorplan[2+(i % (floor_size)), 2 + math.floor(i / (floor_size))] = 'S'
        
        # Rotate the floorplan so it's interpreted as seen in the text file
        floorplan = np.rot90(floorplan, 3)

        # Init params
        self.width = floor_size + 4
        self.height = floor_size + 4
        self.human_count = human_count
        self.visualise_vision = visualise_vision

        # Set up grid
        self.grid = MultiGrid(floor_size + 4, floor_size + 4, torus=False)

        # Used to easily see if a location is a FireExit, since this needs to be done a lot
        self.fire_exits: dict[Coordinate, FireExit] = {}

        # If random spawn is false, spawn_pos_list will contain the list of possible 
        # spawn points according to the floorplan
        self.random_spawn = random_spawn
        self.spawn_pos_list: list[Coordinate] = []

        self.decisioncount = dict()
        self.exitscount = dict()
        
        if not (interact_neumann is None and 
                interact_moore is None and
                interact_swnetwork is None):
            interactionmatrix = {"neumann": interact_neumann, 
                                 "moore": interact_moore,
                                 "swnetwork": interact_swnetwork}
        else:
            interactionmatrix = None 
            
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
                
                "TurnCount": lambda m: self.get_decision_count(self.COUNTER_TURN),
                # add entries for your further decision categories here
                
                "TurnCount": lambda m: self.get_decision_count(self.COUNTER_TURN),
                # add entries for your further decision categories here
                
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
        
        self.G = nx.watts_strogatz_graph(n=self.human_count, k=5, p=0.3, seed = seed)
        self.net = NetworkGrid(self.G)
        nodes = enumerate(self.G.nodes())
                         
        ################################## 
        # Agent creation
        ##################################
        for i in range(0, self.human_count):
            if self.random_spawn:  # Place humans randomly
                pos = tuple(self.rng.choice(tuple(self.grid.empties)))
            else:  # Place humans at specified spawn locations
                pos = tuple(self.rng.choice(self.spawn_pos_list))
                self.spawn_pos_list.remove(pos)
            try:
                # Create a random human
                speed = self.rng.integers(self.MIN_SPEED, self.MAX_SPEED + 1)

                nervousness = -1
                while nervousness < 0 or nervousness > 1:
                    nervousness = self.rng.normal(loc = nervousness_mean, scale = 0.2)
                    
                cooperativeness = -1
                while cooperativeness < 0 or cooperativeness > 1:
                    cooperativeness = self.rng.normal(cooperation_mean, scale=0.1)

                belief_distribution = [alarm_believers_prop, 1 - alarm_believers_prop]
                if interactionmatrix is None:
                    believes_alarm = self.rng.choice([True, False], p=belief_distribution)
                else:
                    believes_alarm = False
                    
                orientation = Human.Orientation(self.rng.integers(1,5))
                    
                if i < (human_count*facilitators_percentage) // 100.0:
                    while nervousness < 0 or nervousness > 1:
                        nervousness = self.rng.normal(loc = nervousness_mean, scale = 0.2)
                    human = Facilitator(
                        speed=speed,
                        orientation = orientation,
                        nervousness = nervousness,
                        cooperativeness=cooperativeness,
                        switches = self.switches,
                        memories = self.agentmemories,
                        memorysize = agentmemorysize,
                        turnwhenblocked_prop = turnwhenblocked_prop,
                        maxsight = maxsight,
                        interactionmatrix = interactionmatrix,
                        model=self,
                        )
                else:
                    while nervousness < 0 or nervousness > 1:
                        nervousness = self.rng.normal(loc = nervousness_mean - 0.3, scale = 0.2)
                    human = Human(
                        speed=speed,
                        orientation=orientation,
                        nervousness=nervousness,
                        cooperativeness=cooperativeness,
                        believes_alarm=believes_alarm,
                        switches = self.switches,
                        memories = self.agentmemories,
                        memorysize = agentmemorysize,
                        maxsight = maxsight,
                        interactionmatrix = interactionmatrix,
                        turnwhenblocked_prop = turnwhenblocked_prop,
                        model=self,
                    )


                self.grid.place_agent(human, pos)
                
                # add to network
                _ , node = next(nodes)
                self.net.place_agent(human, node)
                
            except ValueError:
                print("No tile empty for human placement!")

        # select random agent to propagate alarm
        if interactionmatrix is not None:
            if select_initiator:
                # implement initiator selection here
                initiator = self.rng.choice(self.agents)
            else:
                initiator = self.rng.choice(self.agents)
                
            initiator.believes_alarm = True
        
        self.running = True
        logger.info("Model initialised")

    def step(self):
        """
        Advance the model by one step.
        """

        logger.info("Running step " + str(self.steps))
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)

        # If all humans escaped, stop the model and collect the results
        if self.get_num_escaped(self) == self.human_count:
            self.running = False
        
        # In case there are Humans unable to move let the model run extra steps
        # to indicate the insufficient escape: 
        if self.stepcounter == 0:
            self.running = False
            # final actions:
            # create agent memory
            for agent in self.agents:
                if isinstance(agent, Human):
                    data = pd.DataFrame({'rep': self.modelrun + 1,
                                        'agent':agent.unique_id,
                                        'cooperativeness' : agent.cooperativeness,
                                        'numsteps2escape': agent.numsteps2escape},index = [0])
                    if not self.agentmemories is None:
                        self.agentmemories = pd.concat([self.agentmemories, data])
                    else:
                        self.agentmemories = data
                
        if self.stepcounter >= 0:
            self.stepcounter -=1
        elif self.get_human_speed(self) == 0:
            self.stepcounter = FireEvacuation.EXTRA_STEPS_PER_IMMOBILE *\
                sum(map(lambda agent : isinstance(agent, Human) 
                            and not agent.escaped, self.agents))
                
    def run(self, n):
        """Run the model for n steps."""
        for _ in range(n):
            if self.running or self.stepcounter >= 0:
                self.step()
            
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

    @staticmethod
    def get_num_alarm_believers(model):
        """
        Helper method to count escaped humans
        """
        count = 0
        for agent in model.schedule.agents:
            if isinstance(agent, Human) and agent.believes_alarm == True:
                count += 1

        return count


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

    def escaped(self, pos):
        if pos not in self.exitscount:
            self.exitscount[pos] = 0
        self.exitscount[pos] +=1 


    def get_escaped_exit(self, pos):
        if pos not in self.exitscount.keys():
            return 0
        else:
            return self.exitscount[pos]