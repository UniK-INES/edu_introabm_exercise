from mesa import Model
from mesa.time import BaseScheduler, RandomActivation, SimultaneousActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector
import numpy as np

from .agent import PDAgent


class PdGrid(Model):
    """Model class for iterated, spatial prisoner's dilemma model."""

    schedule_types = {
        "Sequential": BaseScheduler,
        "Random": RandomActivation,
        "Simultaneous": SimultaneousActivation,
    }

    # This dictionary holds the payoff for this agent,
    # keyed on: (my_move, other_move)

    payoff = {("C", "C"): 1, ("C", "D"): 0, ("D", "C"): 1.6, ("D", "D"): 0}

    def __init__(
        self, width=50, height=50, schedule_type="Random", payoffs=None, seed=None,
        printneighbourscore = False, printneighbourorder = False, playfirst = False,
        shuffleagents = False, torus=True, shufflecells = False,
        focalpos = (0,0)
    ):
        """
        Create a new Spatial Prisoners' Dilemma Model.

        Args:
            width, height: Grid size. There will be one agent per grid cell.
            schedule_type: Can be "Sequential", "Random", or "Simultaneous".
                           Determines the agent activation regime.
            payoffs: (optional) Dictionary of (move, neighbor_move) payoffs.
        """
        self.grid = SingleGrid(width, height, torus=torus)
        self.schedule_type = schedule_type
        self.schedule = self.schedule_types[self.schedule_type](self)
        self.printneighbourscore = printneighbourscore
        self.printneighbourorder = printneighbourorder
        self.playfirst = playfirst
        self.shuffleagents = shuffleagents
        self.focalpos = focalpos
        
        self.rng = np.random.default_rng(seed)
        
        xset = list(range(width))
        yset = list(range(height))
        
        if shufflecells:
            self.rng.shuffle(xset)
            self.rng.shuffle(yset)
            
        # Create agents
        for x in xset:
            for y in yset:
                agent = PDAgent((x, y), self)
                self.grid.place_agent(agent, (x, y))
                self.schedule.add(agent)

        self.datacollector = DataCollector(
            {
                "Cooperating_Agents": lambda m: len(
                    [a for a in m.schedule.agents if a.move == "C"]
                )
            }
        )

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        if self.printneighbourscore or self.printneighbourorder:
            print("Step " + str(self.schedule.steps))
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)

    def run(self, n):
        """Run the model for n steps."""
        for _ in range(n):
            self.step()
