from mesa.visualization import SolaraViz, SpaceRenderer, make_space_component, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle
from fire_evacuation.model import FireEvacuation, FireEvacuationScenario
from fire_evacuation.agent import Human, FireExit, Wall, Sight
import os

current_dir = os.path.dirname(os.path.realpath(__file__))

# Specify the parameters changeable by the user, in the web interface
model_params = {
    # "seed":  mesa.visualization.Number
    #      name="Random seed", value=1
    # ),
    "random_spawn": {
        "type": "Checkbox",
        "value": True,
        "label": "Random spawn of initial positions",
    },
    "floor_size": {
        "type": "SliderInt",
        "value": 12,
        "label": "Room size (edge)",
        "min": 5,
        "max": 30,
        "step": 1,
    },
    "human_count": {
        "type": "SliderInt",
        "value": 80,
        "label": "Number Of Human Agents",
        "min": 1,
        "max": 500,
        "step": 5,
    },
    "max_speed": {
        "type": "SliderInt",
        "value": 2,
        "label": "Maximum Speed of agents",
        "min": 1,
        "max": 5,
        "step": 1,
    },
    "alarm_believers_prop": {
        "type": "SliderFloat",
        "value": 1.0,
        "label": "Proportion of Alarm Believers",
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
    },
    "cooperation_mean": {
        "type": "SliderFloat",
        "value": 0.3,
        "label": "Mean Cooperation",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
    },
    "nervousness_mean": {
        "type": "SliderFloat",
        "value": 0.3,
        "label": "Mean Nervousness",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
    },
}

def agent_portrayal(agent):
    size = 55
        
    if type(agent) is Human:

        if agent.believes_alarm:
            # believes in alarm
            shape = os.path.join(current_dir, "fire_evacuation/resources/alarmbeliever.png")
        elif agent.nervousness > Human.NERVOUSNESS_PANIC_THRESHOLD:
            shape = os.path.join(current_dir, "fire_evacuation/resources/panicked_human.png")
        elif agent.humantohelp is not None:
            shape = os.path.join(current_dir, "fire_evacuation/resources/cooperating_human.png")
        else:
            shape = os.path.join(current_dir, "fire_evacuation/resources/human.png")
    elif type(agent) is FireExit:
        shape = os.path.join(current_dir, "fire_evacuation/resources/fire_exit.png")
    elif type(agent) is Wall:
        shape = os.path.join(current_dir, "fire_evacuation/resources/wall.png")
    elif type(agent) is Sight:
        shape = os.path.join(current_dir, "fire_evacuation/resources/eye.png")
    
    # add facilitator portrayal here!
    
    else:
        shape = "X"

    return AgentPortrayalStyle(
        marker=shape,
        size=size,
    )


scenario = FireEvacuationScenario(
        random_spawn = True,
        floor_size = 14,
        human_count = 70,
        alarm_believers_prop = 1.0,
        max_speed = 2,
        seed = 3
)
model = FireEvacuation(scenario)
renderer = SpaceRenderer(model, backend="matplotlib").setup_agents(agent_portrayal)
renderer.render()

page = SolaraViz(
    model,
    model_params = model_params,
    renderer=renderer,
    name="Evacuation Model",
    components=[make_plot_component("AvgNervousness"),
                ],
)

page