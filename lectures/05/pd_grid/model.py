import mesa
import logging
from mesa.discrete_space import OrthogonalMooreGrid
from .agents import PDAgent

logger = logging.getLogger("DPD")

class PdGrid(mesa.Model):
    """Model class for iterated, spatial prisoner's dilemma model."""

    activation_regimes = ["Sequential", "Random", "Simultaneous"]

    # This dictionary holds the payoff for this agent,
    # keyed on: (my_move, other_move)

    payoff = {("C", "C"): 1, ("C", "D"): 0, ("D", "C"): 1.6, ("D", "D"): 0}

    def __init__(
        self, width=50, height=50, activation_order="Random",
        payoffs=None, seed=None,
        printneighbourscore = False,
        printneighbourorder = False,
        initscores = False,
        shuffleneighbors = False,
        torus=True,
        focalpos = (0,0)
    ):
        """
        Create a new Spatial Prisoners' Dilemma Model.

        Args:
            width, height: Grid size. There will be one agent per grid cell.
            activation_order: Can be "Sequential", "Random", or "Simultaneous".
                           Determines the agent activation regime.
            payoffs: (optional) Dictionary of (move, neighbor_move) payoffs.
        """
        super().__init__(seed=seed)
        self.activation_order = activation_order
        self.grid = OrthogonalMooreGrid((width, height), torus=True, random=self.random)
        self.printneighbourscore = printneighbourscore
        self.printneighbourorder = printneighbourorder
        self.shuffleneighbors = shuffleneighbors
        self.focalpos = focalpos
        
        if payoffs is not None:
            self.payoff = payoffs

        PDAgent.create_agents(
            self, len(self.grid.all_cells.cells), cell=self.grid.all_cells.cells
        )

        self.datacollector = mesa.DataCollector(
            {
                "Cooperating_Agents": lambda m: len(
                    [a for a in m.agents if a.move == "C"]
                )
            }
        )

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        if self.printneighbourscore or self.printneighbourorder:
            logger.info("Step " + str(self.schedule.steps))
        # Activate all agents, based on the activation regime
        match self.activation_order:
            case "Sequential":
                self.agents.do("step")
            case "Random":
                self.agents.shuffle_do("step")
            case "Simultaneous":
                self.agents.do("step")
                self.agents.do("advance")
            case _:
                raise ValueError(f"Unknown activation order: {self.activation_order}")

        # Collect data
        self.datacollector.collect(self)

    def run(self, n):
        """Run the model for n steps."""
        for _ in range(n):
            self.step()
        logger.info("Simulation finished. ")
