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
        portrayal["Layer"] = 8
        portrayal["Nervousness"] = agent.nervousness
        portrayal["Cooperation"] = agent.cooperativeness
        portrayal["Believes alarm"] = str(agent.believes_alarm)
        portrayal["Turned"] = agent.turned
        portrayal["Known exits"] = str([exit.pos for exit in agent.exits])
        portrayal["Human2help"] = "None" if agent.humantohelp == None else str(agent.humantohelp.pos)
        portrayal["Target"] = agent.get_planned_target()
        portrayal["Orientation"] = agent.orientation
        portrayal["Vision"] = str(agent.visible_neighborhood)
        portrayal["Speed"] = int(agent.speed)
        portrayal["ID"]= str(agent.unique_id),
        portrayal["text_color"]= "red",
        #if agent.nervousness > Human.NERVOUSNESS_PANIC_THRESHOLD:
            # Panicked
        if agent.believes_alarm:
            portrayal["Shape"] = "fire_evacuation/resources/panicked_human.png"
        elif agent.humantohelp is not None:
            portrayal["Shape"] = "fire_evacuation/resources/cooperating_human.png"
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
        {"Label": "NumEscaped", "Color": "green"},
    ]
)

mobility_chart = ChartModule(
    [
        #{"Label": "Normal", "Color": "green"},
        {"Label": "AvgNervousness", "Color": "red"},
        {"Label": "AvgSpeed", "Color": "green"},
        #{"Label": "Incapacitated", "Color": "blue"},
    ]
)

decision_chart = ChartModule(
    [

        {"Label": "TurnCount", "Color": "red"},
    ]
)

exit_chart = ChartModule(
    [

        {"Label": "EscapedWest", "Color": "red"},
        {"Label": "EscapedSouth", "Color": "blue"},
        {"Label": "EscapedNorth", "Color": "orange"},
        {"Label": "EscapedEast", "Color": "green"},
    ]
)

# Specify the parameters changeable by the user, in the web interface
model_params = {
    "seed": UserSettableParameter(
        "number", "Random seed", value=1
    ),
    "floor_size": UserSettableParameter(
        "slider", "Room size (edge)", value=14, min_value=5, max_value=30, step=1
    ),
    "human_count": UserSettableParameter(
        "slider", "Number Of Human Agents", value=80, min_value=1, max_value=500, step=5
    ),
    "random_spawn": UserSettableParameter(
        "checkbox", "Spawn Agents at Random Locations", value=True
    ),
    "predictcrowd": UserSettableParameter(
        "checkbox", "Let agents predict crowd when turning", value=True
    ),
    "max_speed": UserSettableParameter(
        "slider", "Maximum Speed of agents", value=2, min_value=1, max_value=5, step=1
    ),
    "alarm_believers_prop": UserSettableParameter(
        "slider", "Proportion of Alarm Believers", value=1.0, min_value=0.0, max_value=1.0, step=0.05
    ),
    "cooperation_mean": UserSettableParameter(
        "slider", "Mean Cooperation", value=0.3, min_value=0, max_value=1, step=0.01
    ),
    "nervousness_mean": UserSettableParameter(
        "slider", "Mean Nervousness", value=0.3, min_value=0, max_value=1, step=0.01
    ),
    
    "interact_neumann": UserSettableParameter(
        "slider", "Propagation on vonNeumann", value=0.0, min_value=0, max_value=1, step=0.01
    ), 
    "interact_moore": UserSettableParameter(
        "slider", "Propagation on Moore", value=0.0, min_value=0, max_value=1, step=0.01
    ), 
    "interact_swnetwork": UserSettableParameter(
        "slider", "Propagation on SW-network", value=0.0, min_value=0, max_value=1, step=0.01
    ),   

}

# Start the visual server with the model
server = ModularServer(
    FireEvacuation,
    [canvas_element, status_chart, mobility_chart, decision_chart, exit_chart],
    "Room Evacuation",
    model_params,
)
