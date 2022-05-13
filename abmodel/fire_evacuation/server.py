from os import listdir, path

from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter

from .model import FireEvacuation
from .agent import FireExit, Wall, Human, Sight, Door


# Creates a visual portrayal of our model in the browser interface
def fire_evacuation_portrayal(agent):
    if agent is None:
        return

    portrayal = {}
    (x, y) = agent.get_position()
    portrayal["x"] = x
    portrayal["y"] = y

    if type(agent) is Human:
        portrayal["scale"] = 1
        portrayal["Layer"] = 5
        portrayal["Nervousness"] = agent.nervousness
        portrayal["Speed"] = int(agent.speed)
        portrayal["Health"] = agent.health
        portrayal["text"]= str(agent.unique_id),
        portrayal["text_color"]= "red",
        if agent.get_mobility() == Human.Mobility.INCAPACITATED:
            # Incapacitated
            portrayal["Shape"] = "fire_evacuation/resources/incapacitated_human.png"
            portrayal["Layer"] = 6
        elif agent.get_mobility() == Human.Mobility.PANIC:
            # Panicked
            portrayal["Shape"] = "fire_evacuation/resources/panicked_human.png"
        else:
            # Normal
            portrayal["Shape"] = "fire_evacuation/resources/human.png"
            
    # add facilitator portrayal here!
    
    elif type(agent) is FireExit:
        portrayal["Shape"] = "fire_evacuation/resources/fire_exit.png"
        portrayal["scale"] = 1
        portrayal["Layer"] = 1
    elif type(agent) is Door:
        portrayal["Shape"] = "fire_evacuation/resources/door.png"
        portrayal["scale"] = 1
        portrayal["Layer"] = 1
    elif type(agent) is Wall:
        portrayal["Shape"] = "fire_evacuation/resources/wall.png"
        portrayal["scale"] = 1
        portrayal["Layer"] = 1
    elif type(agent) is Sight:
        portrayal["Shape"] = "fire_evacuation/resources/eye.png"
        portrayal["scale"] = 0.8
        portrayal["Layer"] = 7

    return portrayal

# initial extent that will be changed on reset (?)
canvas_element = CanvasGrid(fire_evacuation_portrayal, 30, 30, 700, 700)

# Define the charts on our web interface visualisation
status_chart = ChartModule(
    [
        {"Label": "Alive in room", "Color": "blue"},
        #{"Label": "Dead", "Color": "red"},
        {"Label": "Escaped", "Color": "green"},
    ]
)

mobility_chart = ChartModule(
    [
        #{"Label": "Normal", "Color": "green"},
        {"Label": "AvgNervousness", "Color": "red"},
        {"Label": "AvgHealth", "Color": "blue"},
        {"Label": "AvgSpeed", "Color": "green"},
        {"Label": "AvgPanicScore", "Color": "orange"},
        #{"Label": "Incapacitated", "Color": "blue"},
    ]
)

# Specify the parameters changeable by the user, in the web interface
model_params = {
    "seed": UserSettableParameter(
        "number", "Random seed", value=1
    ),
    "floor_size": UserSettableParameter(
        "slider", "Room size (edge)", value=12, min_value=5, max_value=30, step=1
    ),
    "human_count": UserSettableParameter(
        "slider", "Number Of Human Agents", value=80, min_value=1, max_value=500, step=5
    ),
    "random_spawn": UserSettableParameter(
        "checkbox", "Spawn Agents at Random Locations", value=True
    ),
    "max_speed": UserSettableParameter(
        "slider", "Maximum Speed of agents", value=2, min_value=1, max_value=5, step=1
    ),
    "alarm_believers_prop": UserSettableParameter(
        "slider", "Proportion of Alarm Believers", value=1.0, min_value=0.0, max_value=1.0, step=0.05
    ),
    "min_health": UserSettableParameter(
        "slider", "Minimum Health", value=0.75, min_value=0.2, max_value=1.0, step=0.05
    ),
    "min_nervousness": UserSettableParameter(
        "slider", "Minimum Nervousness", value=1, min_value=1, max_value=10, step=1
    ),
        
    ## add slider for facilitators_percentage    
}

# Start the visual server with the model
server = ModularServer(
    FireEvacuation,
    [canvas_element, status_chart, mobility_chart],
    "Room Evacuation",
    model_params,
)
