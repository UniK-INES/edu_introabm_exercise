from unittest import TestCase
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from fire_evacuation.model import FireEvacuation, FireEvacuationScenario


class AgentTest(TestCase):
    
    SWITCHES = {
            'PREDICT_CROWD': False,
            'DISTANCE_NOISE': 0.0,
        }
    
    def setUp(self) -> None:        
        scenario = FireEvacuationScenario(
            random_spawn = True,
            floor_size = 10,
            human_count = 64,
            alarm_believers_prop = 1.0,
            max_speed = 2,
            rng = 42,
        )
        self.model = FireEvacuation(scenario)

    def test_get_cowd_level(self):
        agent = self.model.grid.find_nearest_cell((3,3)).agents[0]
        agent.crowdradius = 2
        self.assertEqual(agent._get_crowd_level(), 1.0)
                
        