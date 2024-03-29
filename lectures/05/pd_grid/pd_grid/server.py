from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.UserParam import Choice, Checkbox, NumberInput

from .portrayal import portrayPDAgent
from .model import PdGrid


# Make a world that is 50x50, on a 500x500 display.
canvas_element = CanvasGrid(portrayPDAgent, 50, 50, 500, 500)

model_params = {
    "height": 50,
    "width": 50,
    "seed": NumberInput("Seed", 42),
    "schedule_type": Choice(
        "Scheduler type",
        value="Random",
        choices=list(PdGrid.schedule_types.keys()),
    ),
    "initscores": Checkbox(
        "Init scores?",
        value=True),
}

server = ModularServer(PdGrid, [canvas_element], "Prisoner's Dilemma", model_params)
