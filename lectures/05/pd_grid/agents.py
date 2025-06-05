from mesa.discrete_space import CellAgent
import logging

logger = logging.getLogger("DPD")
logger_neighbours = logging.getLogger("DPD.neighbours")
logger_score = logging.getLogger("DPD.scores")

class PDAgent(CellAgent):
    """Agent member of the iterated, spatial prisoner's dilemma model."""

    def __init__(self, model, starting_move=None, cell=None):
        """
        Create a new Prisoner's Dilemma agent.

        Args:
            model: model instance
            starting_move: If provided, determines the agent's initial state:
                           C(ooperating) or D(efecting). Otherwise, random.
        """
        super().__init__(model)
        self.score = 0
        self.cell = cell
        if starting_move:
            self.move = starting_move
        else:
            self.move = self.random.choice(["C", "D"])
        self.next_move = None
        logger.debug("Initial strategy of {0:2.0f}/{1:2.0f}".format(self.cell.coordinate[0], 
                                                                    self.cell.coordinate[1]) + ": " +
                                                                    str(self.move))

    @property
    def is_cooroperating(self):
        return self.move == "C"

    def step(self):
        """Get the best neighbor's move, and change own move accordingly
        if better than own score."""

        neighbors = [*list(self.cell.neighborhood.agents), self]
        if self.model.shuffleneighbors:
            self.random.shuffle(neighbors)
        best_neighbor = max(neighbors, key=lambda a: a.score)
        self.next_move = best_neighbor.move

        if self.cell.coordinate == self.model.focalpos:
            logger_neighbours.info("Neighbours of {0:2.0f}/{1:2.0f}".format(
                self.cell.coordinate[0], self.cell.coordinate[1]) + ": " + 
                  "".join([agent.move  + "(" + "{:4.1f}".format(agent.score) + 
                           ")-" for agent in neighbors]) + "> " + self.next_move)
        
        if self.cell.coordinate == self.model.focalpos:
            logger_neighbours.info("Neighbours of {0:2.0f}/{1:2.0f}".format(
                self.cell.coordinate[0], self.cell.coordinate[1]) + ": " + 
                  "".join(["({:1.0f}/{:1.0f}".format(agent.cell.coordinate[0], agent.cell.coordinate[1]) + 
                           ") > " for agent in neighbors]))
        
        if self.model.activation_order != "Simultaneous":
            self.advance()

    def advance(self):
        self.move = self.next_move
        self.score += self.increment_score()

    def increment_score(self):
        neighbors = self.cell.neighborhood.agents
        if self.model.shuffleneighbors:
            neighbors = [n for n in neighbors]
            self.random.shuffle(neighbors)
        if self.model.activation_order == "Simultaneous":
            moves = [neighbor.next_move for neighbor in neighbors]
        else:
            moves = [neighbor.move for neighbor in neighbors]

            if self.cell.coordinate == self.model.focalpos:
                logger_score.info("Score of {0:2.0f}/{1:2.0f}: ".format(
                    self.cell.coordinate[0], self.cell.coordinate[1]) + 
                      str(list(self.model.payoff[(self.move, move)] for move in moves)) + 
                      " = " + str(sum(self.model.payoff[(self.move, move)] for move in moves)))
                    
        return sum(self.model.payoff[(self.move, move)] for move in moves)
