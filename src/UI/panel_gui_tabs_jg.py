import autogen
import panel as pn
import openai
import os
import time
import asyncio
from typing import List, Dict
import logging
from src import globals
from src.Models.llm_config import gpt4_config
from src.Agents.agents import agents_dict
from src.Agents.group_chat_manager_agent import CustomGroupChatManager, CustomGroupChat
from src.UI.reactive_chat_jg import ReactiveChat
from src.Agents.agents import avatars
from enum import Enum
from dotenv import load_dotenv

#logging.basicConfig(filename='debug.log', level=logging.DEBUG, 
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

load_dotenv()

logging.basicConfig(level=logging.INFO, 
                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


os.environ["AUTOGEN_USE_DOCKER"] = "False"



##############################################
# Main Adaptive Learning Application
############################################## 
globals.input_future = None
script_dir = os.path.dirname(os.path.abspath(__file__))
progress_file_path = os.path.join(script_dir, '../../progress.json')
   

groupchat = CustomGroupChat(agents=list(agents_dict.values()), 
                              messages=[],
                              max_round=globals.MAX_ROUNDS,
                              send_introductions=True
                              )


manager = CustomGroupChatManager(groupchat=groupchat,
                                filename=progress_file_path, 
                                is_termination_msg=lambda x: x.get("content", "").rstrip().find("TERMINATE") >= 0 )    


# Begin GUI components
reactive_chat = ReactiveChat(agents_dict=agents_dict, avatars=avatars, groupchat_manager=manager)


# Register groupchat_manager and reactive_chat gui interface with ConversableAgents
# Register autogen reply function
# TODO: Consider having each conversible agent register the reply function at init
for agent in groupchat.agents:
    agent.groupchat_manager = manager
    agent.reactive_chat = reactive_chat
    agent.register_reply([autogen.Agent, None], reply_func=agent.autogen_reply_func, config={"callback": None})



#Load chat history on startup
manager.get_chat_history_and_initialize_chat(filename=progress_file_path, chat_interface=reactive_chat.learn_tab_interface) 



# --- Panel Interface ---
def create_app():    
    return reactive_chat.draw_view()

if __name__ == "__main__":    
    app = create_app()
    #pn.serve(app, debug=True)
    pn.serve(app, callback_exception='verbose')
 