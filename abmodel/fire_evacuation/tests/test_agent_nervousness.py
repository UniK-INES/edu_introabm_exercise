from unittest import TestCase, mock, main
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import math

from model import FireEvacuation
from agent import Human, Facilitator

class AgentTest(TestCase):
    
    def setUp(self) -> None:
        print("set up the model and agent here")


    def test_update_nervousness(self):
        assert False