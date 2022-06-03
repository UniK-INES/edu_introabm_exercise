import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time
import math

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import Coordinate, MultiGrid
from mesa.time import RandomActivation

from .agent import Human, Wall, FireExit, Door


class FireEvacuation(Model):
    
    MIN_SPEED = 0
    MAX_SPEED = 3

    COOPERATE_WO_EXIT = False
    
    def __init__(
        self,
        floor_size: int,
        human_count: int,
        visualise_vision = True,
        random_spawn = True,
        alarm_believers_prop = 0.9,
        max_speed = 1,
        cooperation_mean = 0.3,
        nervousness_mean = 0.3,
        cleverturn = False,
        predictcrowd = False,
        agentmemories: pd.DataFrame = None,
        agentmemorysize = 5,
        seed = 1,
     ):
        """
        

        Parameters
        ----------
        floor_size : int
            DESCRIPTION.
        human_count : int
            DESCRIPTION.
        visualise_vision : bool
            DESCRIPTION.
        random_spawn : bool
            DESCRIPTION.
        save_plots : bool
            DESCRIPTION.
         : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
          
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.rngl = np.random.default_rng(seed)
        self.MAX_SPEED = max_speed
        self.COOPERATE_WO_EXIT = FireEvacuation.COOPERATE_WO_EXIT
        
        self.debug = False
        
        self.switches = {
            'CLEVERTURN': cleverturn,
            'PREDICT_CROWD': predictcrowd,
            }
        
        self.stepcounter = -1
        self.agentmemory = agentmemories
        
        if not agentmemories is None:
            self.modelrun = np.max(agentmemories['step'])
        else:
            self.modelrun = -1
        
        # Create floorplan
        floorplan = np.full((floor_size, floor_size), '_')
        floorplan[(0,-1),:]='W'
        floorplan[:,(0,-1)]='W'
        floorplan[math.floor(floor_size/2),(0,-1)] = 'E'
        floorplan[(0,-1), math.floor(floor_size/2)] = 'E'

        # Create floorplan with thicker walls
        floorplan = np.full((floor_size, floor_size), '_')
        floorplan[(0,1,-2,-1),:]='W'
        floorplan[:,(0,1,-2,-1)]='W'
        floorplan[math.floor(floor_size/2),(0,-1)] = 'E'
        floorplan[(0,-1), math.floor(floor_size/2)] = 'E'
        
        floorplan[math.floor(floor_size/2),(1,-2)] = None
        floorplan[(1,-2), math.floor(floor_size/2)] = None
        
        # Rotate the floorplan so it's interpreted as seen in the text file
        floorplan = np.rot90(floorplan, 3)

        # Init params
        self.width = floor_size
        self.height = floor_size
        self.human_count = human_count
        self.visualise_vision = visualise_vision

        # Set up model objects
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(floor_size, floor_size, torus=False)

        # Used to easily see if a location is a FireExit or Door, since this needs to be done a lot
        self.fire_exits: dict[Coordinate, FireExit] = {}
        self.doors: dict[Coordinate, Door] = {}

        # If random spawn is false, spawn_pos_list will contain the list of possible 
        # spawn points according to the floorplan
        self.random_spawn = random_spawn
        self.spawn_pos_list: list[Coordinate] = []

        self.decisioncount = dict()
        self.exitscount = dict()
        
        # Load floorplan objects
        for (x, y), value in np.ndenumerate(floorplan):
            pos: Coordinate = (x, y)

            value = str(value)
            floor_object = None
            if value == "W":
                floor_object = Wall(pos, self)
            elif value == "E":
                floor_object = FireExit(pos, self)
                self.fire_exits[pos] = floor_object
                # Add fire exits to doors as well, since, well, they are
                self.doors[pos] = floor_object
            elif value == "D":
                floor_object = Door(pos, self)
                self.doors[pos] = floor_object
            elif value == "S":
                self.spawn_pos_list.append(pos)
            if floor_object:
                self.grid.place_agent(floor_object, pos)
                self.schedule.add(floor_object)

        # Create a graph of traversable routes, used by humans for pathing
        self.graph = nx.Graph()
        for agents, x, y in self.grid.coord_iter():
            pos = (x, y)

            # If the location is empty, or there are no non-traversable humans
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
                "AvgNervousness": lambda m: self.get_human_nervousness(m)/10,
                "AvgSpeed": lambda m: self.get_human_speed(m),
                
                
                "TurnCount": lambda m: self.get_decision_count(m, "TURN"),
                
                "EscapedWest": lambda m: self.get_escaped_exit(m, list(self.fire_exits)[0]),
                "EscapedSouth": lambda m: self.get_escaped_exit(m, list(self.fire_exits)[1]),
                "EscapedNorth": lambda m: self.get_escaped_exit(m, list(self.fire_exits)[2]),
                "EscapedEast": lambda m: self.get_escaped_exit(m, list(self.fire_exits)[3]),
             }
        )
        
        # Start placing human humans
        for i in range(0, self.human_count):
            if self.random_spawn:  # Place human humans randomly
                pos = tuple(self.rng.choice(tuple(self.grid.empties)))
            else:  # Place human humans at specified spawn locations
                pos = self.rng.choice(self.spawn_pos_list)

            if pos:
                # Create a random human
                speed = self.rng.integers(self.MIN_SPEED, self.MAX_SPEED + 1)

                nervousness = -1
                while nervousness < 0 or nervousness > 1:
                    nervousness = self.rng.normal(nervousness_mean)

                
                cooperativeness = -1
                while cooperativeness < 0 or cooperativeness > 1:
                    cooperativeness = self.rng.normal(cooperation_mean)

                belief_distribution = [alarm_believers_prop, 1 - alarm_believers_prop]
                believes_alarm = self.rng.choice([True, False], p=belief_distribution)

                orientation = Human.Orientation(self.rng.integers(1,5))
                
                # decide here whether to add a facilitator
                
                if not self.agentmemory is None:
                    memory = self.agentmemory[self.agentmemory['agent']==i]
                else:
                    memory = None
                    
                human = Human(
                    i,
                    pos,
                    speed=speed,
                    orientation=orientation,
                    nervousness=nervousness,
                    cooperativeness=cooperativeness,
                    believes_alarm=believes_alarm,
                    switches = self.switches,
                    model=self,
                    memory = memory,
                    memorysize = agentmemorysize
                )

                self.grid.place_agent(human, pos)
                self.schedule.add(human)
            else:
                print("No tile empty for human placement!")

        self.running = True

    def step(self):
        """
        Advance the model by one step.
        """

        self.schedule.step()
        self.datacollector.collect(self)

        # If all humans escaped, stop the model and collect the results
        if self.get_num_escaped(self) == self.human_count:
            self.running = False
        
        if self.stepcounter == 0:
            self.running = False
            # final actions:
            # create agent memory
            for agent in self.schedule.agents:
                if isinstance(agent, Human):
                    data = pd.DataFrame({'step': self.modelrun + 1,
                                                      'agent':agent.unique_id,
                                                      'cooperativeness' : agent.cooperativeness,
                                                      'numsteps2escape': self.schedule.steps},index = [0])
                                                      #'numsteps2escape': agent.numsteps2escape},index = [0])
                    if not self.agentmemory is None:
                        self.agentmemory = pd.concat([self.agentmemory, data])
                    else:
                        self.agentmemory = data
                
        if self.stepcounter >= 0:
            self.stepcounter -=1
        elif self.get_human_speed(self) == 0:
            self.stepcounter = 10 * sum(map(lambda agent : isinstance(agent, Human) and not agent.escaped, self.schedule.agents))
                
    def run(self, n):
        """Run the model for n steps."""
        for _ in range(n):
            if self.running or self.stepcounter >= 0:
                self.step()
            
    def get_agentmemories(self):
        return self.agentmemory
       
    @staticmethod     
    def get_human_nervousness(model):
        count = 0
        nervousness = 0
        for agent in model.schedule.agents:
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
        for agent in model.schedule.agents:
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
        for agent in model.schedule.agents:
            if isinstance(agent, Human) and agent.escaped == True:
                count += 1

        return count
    
    @staticmethod
    def escaped(model, pos):
        if pos not in model.exitscount:
            model.exitscount[pos] = 0
        model.exitscount[pos] +=1 
        
    @staticmethod
    def get_escaped_exit(model, pos):
        if pos not in model.exitscount.keys():
            return 0
        else:
            return model.exitscount[pos]
    
    @staticmethod
    def increment_decision_count(model, decision):
        if decision not in model.decisioncount:
            model.decisioncount[decision] = 0
        model.decisioncount[decision] +=1 
    
    @staticmethod
    def get_decision_count(model, decision):
        return model.decisioncount[decision]
