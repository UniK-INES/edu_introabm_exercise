from mesa.discrete_space.cell import Cell, Coordinate
from mesa.discrete_space.cell_agent import CellAgent, FixedAgent
from mesa import Agent
import networkx as nx
from enum import IntEnum
import math


def get_line(start, end):
    """
    Implementation of Bresenham's Line Algorithm
    Returns a list of tuple coordinates from starting tuple to end tuple (and including them)
    """
    # Break down start and end tuples
    x1, y1 = start
    x2, y2 = end

    # Calculate differences
    diff_x = x2 - x1
    diff_y = y2 - y1

    # Check if the line is steep
    line_is_steep = abs(diff_y) > abs(diff_x)

    # If the line is steep, rotate it
    if line_is_steep:
        # Swap x and y values for each pair
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # If the start point is further along the x-axis than the end point, swap start and end
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Calculate the differences again
    diff_x = x2 - x1
    diff_y = y2 - y1

    # Calculate the error margin
    error_margin = int(diff_x / 2.0)
    step_y = 1 if y1 < y2 else -1

    # Iterate over the bounding box, generating coordinates between the start and end coordinates
    y = y1
    path = []

    for x in range(x1, x2 + 1):
        # Get a coordinate according to if x and y values were swapped
        coord = (y, x) if line_is_steep else (x, y)
        path.append(coord)  # Add it to our path
        # Deduct the absolute difference of y values from our error_margin
        error_margin -= abs(diff_y)

        # When the error margin drops below zero, increase y by the step and the error_margin by the x difference
        if error_margin < 0:
            y += step_y
            error_margin += diff_x

    # The the start and end were swapped, reverse the path
    if swapped:
        path.reverse()

    return path


"""
FLOOR STUFF
"""

class FloorObject(FixedAgent):
    def __init__(
        self,
        traversable: bool,
        visibility: int = 2,
        model=None,
    ):
        super().__init__(model)
        self.traversable = traversable
        self.visibility = visibility


class Sight(FloorObject):
    def __init__(self, model):
        super().__init__(
            traversable=True, visibility=-1, model=model
        )

class FireExit(FloorObject):
    def __init__(self, model):
        super().__init__(
            traversable=True, visibility=6, model=model)


class Wall(FloorObject):
    def __init__(self, model):
        super().__init__(traversable=False, model=model)


class Furniture(FloorObject):
    
    def __init__(self, model):
        super().__init__(traversable=False, model=model)


class Human(CellAgent):
    """
    A human agent, which will attempt to escape from the grid.

    Attributes:
        ID: Unique identifier of the Agent
        Position (x,y): Position of the agent on the Grid
        Health: Health of the agent (between 0 and 1)
        ...
    """
        
    class Orientation(IntEnum):
        NORTH = 1
        EAST = 2
        SOUTH = 3
        WEST = 4
    
    MIN_SPEED = 0
    MAX_SPEED = 3

    CROWD_RADIUS = 3
    
    CROWD_RELAXATION_THRESHOLD = 0.6
    CROWD_ANXIETY_THRESHOLD = 0.8
    
    CROWD_ANXIETY_INCREASE = 0.2
    CROWD_RELAXATION_DECREASE = 0.1
    
    NERVOUSNESS_SPEEDCHANGE = 1
    NERVOUSNESS_DECREASE_HELP = 0.5
    NERVOUSNESS_INCREASE_BLOCKED = 0.2

    SPEED_RECOVERY_PROBABILTY = 0.25
    
    # The value the nervousness score must reach for an agent to start panic behaviour
    NERVOUSNESS_PANIC_THRESHOLD = 0.8
    NERVOUSNESS_SPEEDCHANGE_THRESHOLD = 0.7
    RANDOMWALK_PROB = 0.3
    
    COOPERATIVENESS_THRESHOLD = 0.5
    
        
    def __init__(self,
            speed: int,
            orientation: Orientation,
            nervousness: float,
            cooperativeness: float,
            believes_alarm: bool,
            model,
            turnwhenblocked_prop: float,
        ):
        
        """
        Init agent properties.

        Parameters
        ----------
        speed : int
            number of tiles to go during a simulation step
            
        orientation: Orientation
            initial orientation of the agent (NORTH, EAST, SOUTH, WEST)
            
        nervousness: float
            value 0...1
            
        cooperativeness: float
            value 0...1
            
        believes_alarm: bool
        
        model: Model
            model

        Returns
        -------
        None.

        """
        
        super().__init__(model)

        ''' Human humans should not be traversable, but we allow 
        "displacement", e.g. pushing to the side'''
        self.traversable = False
        self.orientation = orientation
        self.speed: int = speed
        self.crowdradius = Human.CROWD_RADIUS
        self.nervousness = nervousness
        self.turnwhenblocked_prop = turnwhenblocked_prop
        self.cooperativeness = cooperativeness
        # Boolean stating whether or not the agent believes the alarm is a real fire
        self.believes_alarm = believes_alarm
        self.turned = False  
        self.escaped: bool = False
        
        self.visible_neighborhood = set()
        self.exits = dict()
        self.humans = dict()
        self.humantohelp = None
        
        # The agent and seen location (agent, (x, y)) the agent is planning to move to
        self.planned_target: Agent = None

        self.visible_tiles: tuple[Coordinate, tuple[Agent]] = []
        self.knownExits: tuple[Coordinate] = [] 


    def step(self):
        """
        A single step of the human agent.
        """
        if not self.escaped and self.cell:
            self.turned = False
            
            ######################
            # Update properties
            ######################

            self.update_nervousness()
            self.learn_fieldofvision()
            self.update_speed()
            
            ######################
            # Decide action:
            ######################

            # check panic mode
            if self.nervousness > Human.NERVOUSNESS_PANIC_THRESHOLD:
                if self.model.rng.random() < Human.RANDOMWALK_PROB:
                    self.get_random_target()
            
            else:        
                # check cooperation
                if self.cooperativeness > self.COOPERATIVENESS_THRESHOLD and self.humantohelp == None \
                        and (len(self.exits) > 0 
                        or self.model.COOPERATE_WO_EXIT):
                    self.cooperate()
                        
                # If the agent believes the alarm, attempt to plan 
                # an exit location if we haven't already and we aren't performing an action
                if not self.turned and not isinstance(self.planned_target, FireExit) and not isinstance(self.planned_target, Human):
                    if self.believes_alarm:
                        self.attempt_exit_plan()

            ######################
            # Perform action:
            ######################
            
            if not self.turned:
                if self.planned_target == None:
                    self.get_random_target()
                
                    
                # finally go
                self.move_toward_target()
                
                self.help()
    
                # Agent reached a fire escape, proceed to exit
                if self.cell in self.model.fire_exits.keys():
                    self.escaped = True
                    self.remove()
 
    def update_nervousness(self):
        """
        Update the humans' nervousness based on crowd level.
        """
        crowdlevel = self._get_crowd_level()
        if crowdlevel > Human.CROWD_ANXIETY_THRESHOLD:
            self.nervousness += Human.CROWD_ANXIETY_INCREASE
        elif crowdlevel < Human.CROWD_RELAXATION_THRESHOLD:
            self.nervousness -= Human.CROWD_RELAXATION_DECREASE
        self.nervousness = min(max(0.0, self.nervousness), 1.0) 


    def _get_crowd_level(self):
        agentcounter = 0 
        neighbourhood = self.cell.get_neighborhood(
                radius = self.crowdradius)   
        for agent in neighbourhood:
            if isinstance(agent, Human):
                agentcounter +=1
        return agentcounter / len(neighbourhood)    


    def learn_fieldofvision(self):
        """
        Perceive environment in the direction of the agent's orientation.
        Add found agents and exits to agent's internal memory.
        """
        self.visible_neighborhood = self._explore_fieldofvision(self.orientation)
        self.humans = dict()
        
        # add agents in found cells
        for agent in [agent
                      for cell in self.visible_neighborhood
                      for agent in cell.agents]:
            if isinstance(agent, FireExit):
                self.exits[agent]=None
            elif isinstance(agent, Human):
                self.humans[agent]=None


    def _explore_fieldofvision(self, orientation):
        visible_neighborhood = list()

        # gather cells in a 90° angle in the human's direction:
        if orientation == Human.Orientation.NORTH:
            startx = stopx = self.cell.coordinate[0]
            for y in range(self.cell.coordinate[1] + 1, self.model.grid.height):
                startx = max(startx-1, 0)
                stopx = min(stopx + 1, self.model.grid.width)
                for x in range(startx, stopx):
                    visible_neighborhood.append(self.model.grid.find_nearest_cell([x,y]))
        elif orientation == Human.Orientation.SOUTH:
            startx = stopx = self.cell.coordinate[0]
            for y in range(self.cell.coordinate[1] - 1, -1, -1):
                startx = max(startx-1, 0)
                stopx = min(stopx + 1, self.model.grid.width)
                for x in range(startx, stopx):
                    visible_neighborhood.append(self.model.grid.find_nearest_cell([x,y]))
        elif orientation == Human.Orientation.WEST:
            starty = stopy = self.cell.coordinate[1]
            for x in range(self.cell.coordinate[0] - 1, -1, -1):
                starty = max(starty-1, 0)
                stopy = min(stopy + 1, self.model.grid.height)
                for y in range(starty, stopy):
                    visible_neighborhood.append(self.model.grid.find_nearest_cell([x,y]))                          
        elif orientation == Human.Orientation.EAST:
            starty = stopy = self.cell.coordinate[1]
            for x in range(self.cell.coordinate[0] + 1, self.model.width):
                starty = max(starty-1, 0)
                stopy = min(stopy + 1, self.model.grid.height)
                for y in range(starty, stopy):
                    visible_neighborhood.append(self.model.grid.find_nearest_cell([x,y]))
        return visible_neighborhood


    def update_speed(self):
        """
        When the human's nervousness exceeds a threshold, it's speed  
        is either slowed down or accelerated (panic situation).
        With probability Human.SPEED_RECOVERY_PROBABILTY frozen humans
        start to move slowly again.
        """
        if self.nervousness > Human.NERVOUSNESS_SPEEDCHANGE_THRESHOLD:
            self.speed = int(min(max(Human.MIN_SPEED, 
                                     self.speed + self.model.rng.choice([-1, 1])),
                                     Human.MAX_SPEED)) 
        
        if self.speed == 0 and self.model.rng.random() < Human.SPEED_RECOVERY_PROBABILTY:
            self.speed = 1


    def get_random_target(self):
        """
        Choose random cell
        """
        # exclude walls!
        x = self.model.rng.integers(2, self.model.grid.width - 2)
        y = self.model.rng.integers(2, self.model.grid.height - 2)
        self.planned_target = Agent(self.model)
        self.planned_target.cell = self.model.grid.find_nearest_cell([x,y])

      
    def cooperate(self):
        """
        Find close-by human in need of help.
        Criteria: speed = 0, not believing alarm, no exit target
        """
        if len(self.humans) > 0:
            distance = math.inf
            closebyhuman = None
            for human in self.humans.keys():
                if human.speed == 0 or human.believes_alarm == False or len(human.exits) == 0:
                    curdist = self._get_euclidean_distance(human.cell)
                    if curdist < distance:
                        distance = curdist
                        closebyhuman = human
            
            if not closebyhuman == None:
                self.planned_target = closebyhuman
                self.humantohelp = closebyhuman


    def _get_euclidean_distance(self, cell):
        return math.sqrt(abs(self.cell.coordinate[0] - cell.coordinate[0])**2 + 
                         abs(self.cell.coordinate[1] - cell.coordinate[1])**2)


    def attempt_exit_plan(self):
        """
        Find a target to exit.
        """
        self.planned_target = None

        if len(self.exits) > 0:
            if len(self.exits) > 1:  
                # If there is more than one exit known
                best_distance = None
                for exitdoor in self.exits.keys():
                    # Let's use Bresenham's to find the 'closest' exit
                    length = len(get_line(self.cell.coordinate, exitdoor.cell.coordinate))
                    if not best_distance or length < best_distance:
                        best_distance = length
                        self.planned_target = exitdoor

            else:
                self.planned_target = list(self.exits.keys())[0]

        elif self.turned == False:
            # If there's no fire-escape in sight, turn around
            self.turn()            


    def turn(self):
        """
        Turn clockwise.
        """
        self.orientation = Human.Orientation(self.orientation % 4 + 1 )
        self.turned = True


    def get_path(self, graph, target, include_target=True) -> list[Coordinate]:
        """
        Get path to target from graph

        Parameters
        ----------
        graph : nx graph
            graph of traversable ways over the floor plan.
        target : tile
            target tile
        include_target : bool, optional
            The default is True.

        Raises
        ------
        Exception
            Current position not found

        Returns
        -------
        list[Coordinate]
            an empty path if no path can be found

        """
        path = []

        try:
            path = nx.shortest_path(graph, self.cell, target)
            if not include_target:
                del path[
                    -1
                ]  # We don't want the target included in the path, so delete the last element

            return list(path)
        except nx.exception.NodeNotFound as e:
            graph_nodes = graph.nodes()

            if target not in graph_nodes:
                contents = target.agents
                print(f"Target node not found! Expected {target}, with contents {contents}")
                return path
            elif self.cell not in graph_nodes:
                contents = self.cell.agents
                raise Exception(
                    f"Current position not found!\nPosition: {self.cell},\nContents: {contents}"
                )
            else:
                raise e

        except nx.exception.NetworkXNoPath as e:
            return path


    def _location_is_traversable(self, cell) -> bool:
        if not cell.is_empty:
            for agent in cell.agents:
                if not agent.traversable:
                    return False
        return True


    def move_toward_target(self):
        """
        The human tries to move towards its target.
        """
        next_location: Cell = None
        pruned_edges = set()
        graph = self.model.graph

        while self.planned_target.cell and not next_location:
            path = self.get_path(graph, self.planned_target.cell)
            
            if isinstance(self.planned_target, Human):
                # to help a human, the agent needs to be next to the human
                path = path[0:-1]

            if len(path) > 0:
                next_location, _ = self._get_next_location(path)

                if next_location == self.cell:
                    continue

                if next_location == None:
                    raise Exception("Next location can't be none")

                # Test the next location to see if we can move there
                if self._location_is_traversable(next_location):
                    # Move normally
                    self.previous_cell = self.cell
                    self.cell = next_location
                    
                elif self.cell == path[-1]:
                    # The human reached their target!
                   
                    self.planned_target = None
                    break

                else:
                    # We want to move here but it's blocked

                    # check if the location is blocked due to a Human agent
                    pushed = False
                    for agent in next_location.agents:
                        # Test the panic value to see if this agent "pushes" the 
                        # blocking agent aside
                        if isinstance(agent, Human):
                            if self.nervousness >= Human.NERVOUSNESS_PANIC_THRESHOLD:
                                # push the agent and then move to the next_location
                                self._push_human_agent(agent)
                                self.previous_cell = self.cell
                                self.cell = next_location
                                pushed = True
                                break
                            elif self.model.rng.random() < self.turnwhenblocked_prop:
                                self.turn()
                                break
                    if self.turned:
                        break
                    if pushed:
                        continue

                    # Remove the next location from the temporary graph so we 
                    # can try pathing again without it
                    edges = graph.edges(next_location)
                    pruned_edges.update(edges)
                    graph.remove_node(next_location)

                    # Reset planned_target if the next location was the end of the path
                    if next_location == path[-1]:
                        next_location = None
                        self.planned_target = None
                        break
                    else:
                        next_location = None

            else:  # No path is possible, so drop the target
                self.planned_target = None
                self.nervousness += Human.NERVOUSNESS_INCREASE_BLOCKED
                break

        if len(pruned_edges) > 0:
            # TODO does not seem to be necessary, as graph is not used after this in this function
            # Add back the edges we removed when removing any non-traversable nodes 
            # from the global graph, because they may be traversable again next step
            graph.add_edges_from(list(pruned_edges))


    def _get_next_location(self, path):
        """
        Extract the path and target for the next tick.

        Parameters
        ----------
        path : tuple
            currently followed path.

        Raises
        ------
        Exception
            Failure when determining next location.

        Returns
        -------
        next_location : Cell
            Next location to end at.
        next_path : tuple
            Path to next location.

        """
        path_length = len(path)

        try:
            if path_length <= self.speed:
                next_location = path[path_length - 1]
            else:
                next_location = path[self.speed]

            next_path = []
            for location in path:
                next_path.append(location)
                if location == next_location:
                    break

            return (next_location, next_path)
        except Exception as e:
            raise Exception(
                f"Failed to get next location: {e}\nPath: {path},\nlen: {path_length},\nSpeed: {self.speed}"
            )


    def _push_human_agent(self, agent):
        """
        Pushes the agent to a neighbouring tile

        Parameters
        ----------
        agent
            agent to push.
        """
        neighborhood = self.cell.get_neighborhood(
            radius=1,
            include_center=False,
        )
        traversable_neighborhood = [
            neighbor_cell
            for neighbor_cell in neighborhood
            if self._location_is_traversable(neighbor_cell)
        ]

        if len(traversable_neighborhood) > 0:
            # push the human agent to a random traversable position
            i = self.rng.choice(len(traversable_neighborhood))
            push_cell = traversable_neighborhood[i]
            agent.cell = push_cell
            agent.nervousness += 0.1


    def help(self):
        """
        Help another human in need of support to
        - calm down
        - move again
        - believe the alarm
        - show exits.
        """
        if self.humantohelp != None:
            if self.humantohelp.escaped:
                self.humantohelp = None
                self.planned_target = None
            # reached human to help?
            elif self.humantohelp in self.cell.get_neighborhood():
                self.humantohelp.nervousness -= Human.NERVOUSNESS_DECREASE_HELP
                self.humantohelp.nervousness = min(max(0.0, self.humantohelp.nervousness), 1.0) 
                if self.humantohelp.speed == 0:
                    self.humantohelp.speed = 1
                elif not self.humantohelp.believes_alarm:
                    self.humantohelp.believes_alarm = True
                elif len(self.exits) > 0:
                    self.humantohelp.exits = self.exits
                self.humantohelp = None
                self.planned_target = None


    def get_speed(self):
        return self.speed

    
    def get_planned_target(self):
        if self.planned_target != None:
            return str(self.planned_target.cell)
        else:
            return "none"


    def set_believes(self, value: bool):
        if value and not self.believes_alarm:
            self.believes_alarm = value

            
# Add the new Facilitator class here!
