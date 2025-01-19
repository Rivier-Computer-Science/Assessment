#Import necessary modules for agent functionality
import autogen
import os
from .base_agent import MyBaseAgent
from .conversable_agent import MyConversableAgent
from .interaction_agent import InteractionAgent
from .tutor_agent import TutorAgent

from src.Models.llm_config import gpt4_config
from enum import Enum
#set environment variable to disable Docker for Autogen usage
os.environ["AUTOGEN_USE_DOCKER"] = "False"

###############################################
# ChatGPT Model
###############################################
llm = gpt4_config         #Set up the GPT-4 configuration for the model

#################################################
# Define Agents
#################################################
#Create instances of InteractionAgent and TutorAgent, passing the GPT-4 configuration
interaction_agent = InteractionAgent(llm_config=llm)        # Define the agent for user interactions.
tutor = TutorAgent(llm_config=llm)         # Define the tutor agent, likely for learning sessions.


# Enum to map agent types (e.g., tutor, interaction)
class AgentKeys(Enum):
    TUTOR = 'tutor'        # Represents tutor agent.
    INTERACTION = 'interaction'        # Represents interaction agent.

# Agents
interaction = InteractionAgent()         
tutor = TutorAgent()                


agents_dict = {
    AgentKeys.INTERACTION.value: interaction,      # Associate interaction agent with its key.   
    AgentKeys.TUTOR.value: tutor,                # Associate tutor agent with its key.
 }

agents_dict_by_name = {
    "InteractionAgent": interaction,          # Map the string name of the agent to the instance.
    "TutorAgent": tutor,                    # Map the string name of the tutor agent to the instance.
}

avatars = {
    interaction.name: "✏️",                 # Pencil
    tutor.name: "🧑‍🎓",                  # Person with graduation hat
}
