
from dotenv import load_dotenv
load_dotenv()

import autogen
import panel as pn
import openai
import os
import time
import asyncio
from typing import List, Dict
import logging
from src import globals as globals
#from src.Agents.agents import agents_dict
from src.Agents.group_chat_manager_agent import CustomGroupChatManager, CustomGroupChat
from src.UI.avatar import avatar
from enum import Enum

# Telugu specific
from src.UI.reactive_chat24_telegu import ReactiveChat
from src.FSMs.fsm_telugu import TeachMeFSM


#logging.basicConfig(filename='debug.log', level=logging.DEBUG, 
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logging.basicConfig(level=logging.INFO, 
                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

os.environ["AUTOGEN_USE_DOCKER"] = "False"

###############################################
# ChatGPT Model
###############################################

gpt4_config_list = [
    {
        'model': "gpt-4o",
    }
]
# These parameters attempt to produce precise reproducible results
temperature = 0
max_tokens = 1000
top_p = 0.5
frequency_penalty = 0.1
presence_penalty = 0.1
seed = 53


gpt4_config = {"config_list": gpt4_config_list, 
               "temperature": temperature,
               "max_tokens": max_tokens,
               "top_p": top_p,
               "frequency_penalty": frequency_penalty,
               "presence_penalty": presence_penalty,
               "seed": seed
}

llm = gpt4_config

#################################################
# Define Agents
#################################################
from src.Agents.student_agent import StudentAgent
from src.Agents.telugu_teaching_agent import TeluguTeachingAgent
from src.Agents.tutor_agent import TutorAgent
from src.Agents.problem_generator_agent import ProblemGeneratorAgent
from src.Agents.solution_verifier_agent import SolutionVerifierAgent
from src.Agents.learner_model_agent import LearnerModelAgent
from src.Agents.level_adapter_agent import LevelAdapterAgent
from src.Agents.motivator_agent import MotivatorAgent

class AgentKeys(Enum):
    TEACHER = 'teacher'
    TUTOR = 'tutor'
    STUDENT = 'student'
    PROBLEM_GENERATOR = 'problem_generator'
    SOLUTION_VERIFIER = 'solution_verifier'
    LEARNER_MODEL = 'learner_model'
    LEVEL_ADAPTER = 'level_adapter'
    MOTIVATOR = 'motivator'
 
# Agents
student = StudentAgent(llm_config=llm)
teacher = TeluguTeachingAgent(llm_config=llm)
tutor = TutorAgent(llm_config=llm)
problem_generator = ProblemGeneratorAgent(llm_config=llm)
solution_verifier = SolutionVerifierAgent(llm_config=llm)
learner_model = LearnerModelAgent(llm_config=llm)
level_adapter = LevelAdapterAgent(llm_config=llm)
motivator = MotivatorAgent(llm_config=llm)

agents_dict = {
    AgentKeys.STUDENT.value: student,
    AgentKeys.TEACHER.value: teacher,
    AgentKeys.TUTOR.value: tutor,
    AgentKeys.PROBLEM_GENERATOR.value: problem_generator,
    AgentKeys.SOLUTION_VERIFIER.value: solution_verifier,
    AgentKeys.LEARNER_MODEL.value: learner_model,
    AgentKeys.LEVEL_ADAPTER.value: level_adapter,
    AgentKeys.MOTIVATOR.value: motivator,
}

avatars = {
    student.name: "✏️",                 # Pencil
    teacher.name: "🧑‍🎓" ,                # Female teacher
    tutor.name: "👩‍🏫",                  # Person with graduation hat
    problem_generator.name: "📚",  # Stack of books for problem generation
    solution_verifier.name: "🔍",  # Magnifying glass for solution verification
    learner_model.name: "🧠",      # Brain emoji for learner model
    level_adapter.name: "📈",      # Chart with upwards trend for level adaptation
    motivator.name: "🏆",  
 }

##############################################
# Main Adaptive Learning Application
############################################## 
globals.input_future = None
script_dir = os.path.dirname(os.path.abspath(__file__))
progress_file_path = os.path.join(script_dir, '../../progress.json')

logging.debug("Initializing TeachMeFSM")   
fsm = TeachMeFSM(agents_dict)
logging.debug("TeachMeFSM initialized")

groupchat = CustomGroupChat(agents=list(agents_dict.values()), 
                              messages=[],
                              max_round=globals.MAX_ROUNDS,
                              send_introductions=True,
                              speaker_selection_method=fsm.next_speaker_selector
                              )


manager = CustomGroupChatManager(groupchat=groupchat,
                                filename=progress_file_path, 
                                is_termination_msg=lambda x: x.get("content", "").rstrip().find("TERMINATE") >= 0 )    

# Allow the fsm to get the groupchat history
fsm.register_groupchat_manager(manager)
logging.debug("fsm registered groupchat_manager")

# Begin GUI components
reactive_chat = ReactiveChat(agents_dict=agents_dict, avatars=avatars, 
                             groupchat_manager=manager)


# Register groupchat_manager and reactive_chat gui interface with ConversableAgents
# Register autogen reply function
# TODO: Consider having each conversible agent register the reply function at init
for agent in groupchat.agents:
    agent.groupchat_manager = manager
    agent.reactive_chat = reactive_chat
    agent.register_reply([autogen.Agent, None], reply_func=agent.autogen_reply_func, config={"callback": None})



#Load chat history on startup
manager.get_chat_history_and_initialize_chat(
    initial_message="Welcome to the Telugu Teacher! How can I help you today?",
    avatars=avatars,
    filename=progress_file_path, 
    chat_interface=reactive_chat.learn_tab_interface) 

reactive_chat.update_dashboard()    #Call after history loaded



# --- Panel Interface ---
def create_app():    
    return reactive_chat.draw_view()

if __name__ == "__main__":    
    app = create_app()
    #pn.serve(app, debug=True)
    pn.serve(app, callback_exception='verbose')