from mesa.visualization import SolaraViz, SpaceRenderer, make_space_component, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle
from fire_evacuation.model import FireEvacuation, FireEvacuationScenario
from fire_evacuation.agent import Human, FireExit, Wall, Sight, Facilitator
import os
import solara

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
    "facilitators_percentage": {
        "type": "SliderFloat",
        "value": 20,
        "label": "Percentage of Facilitators",
        "min": 0.0,
        "max": 100.0,
        "step": 1.0,
    },
}

def agent_portrayal(agent):
    size = 55
    
    tooltip = {}
        
    if isinstance(agent,Human):
        
        tooltip["Nervousness"] = agent.nervousness
        tooltip["Cooperation"] = agent.cooperativeness
        tooltip["Believes alarm"] = str(agent.believes_alarm)
        tooltip["Turned"] = agent.turned
        tooltip["Known exits"] = str(agent.exits)
        tooltip["Target"] = agent.get_planned_target()
        tooltip["Orientation"] = agent.orientation
        tooltip["Vision"] = str(agent.visible_neighborhood)
        tooltip["Speed"] = int(agent.speed)
        tooltip["ID"]= str(agent.unique_id),
    
        if agent.nervousness > Human.NERVOUSNESS_PANIC_THRESHOLD:
            shape = os.path.join(current_dir, "resources/panicked_human.png")
        elif agent.humantohelp is not None:
            shape = os.path.join(current_dir, "resources/cooperating_human.png")
        elif type(agent) is Facilitator:
            shape = os.path.join(current_dir, "resources/facilitator.png")
        elif agent.believes_alarm:
            # believes in alarm
            shape = os.path.join(current_dir, "resources/alarmbeliever.png")
        else:
            shape = os.path.join(current_dir, "resources/human.png")
    elif type(agent) is FireExit:
        shape = os.path.join(current_dir, "resources/fire_exit.png")
    elif type(agent) is Wall:
        shape = os.path.join(current_dir, "resources/wall.png")
    elif type(agent) is Sight:
        shape = os.path.join(current_dir, "resources/eye.png")
    else:
        shape = "X"

    return AgentPortrayalStyle(
        color="red",
        marker=shape,
        size=size,
        # tooltip=tooltip, # only with altair, which currently does not render PNGs correctly
    )


scenario = FireEvacuationScenario(
        random_spawn = True,
        floor_size = 14,
        human_count = 70,
        alarm_believers_prop = 0.5,
        nervousness_mean = 0.3,
        max_speed = 2,
        seed = 3
)
model = FireEvacuation(scenario)
renderer = SpaceRenderer(model, backend="altair").setup_agents(agent_portrayal)
renderer.render()

page = SolaraViz(
    model,
    model_params = model_params,
    renderer=renderer,
    name="Evacuation Model",
    components=[#make_space_component(agent_portrayal),
                make_plot_component("AvgNervousness",
                                    "TurnCount",
                                    ),
                ],
)

page