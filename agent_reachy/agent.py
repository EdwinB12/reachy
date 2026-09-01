from langchain.agents import create_agent
from reachy_mini import ReachyMini

from agent_reachy.model import Model
from agent_reachy.tools import build_toolbox
from agent_reachy.system_prompt import SYSTEM_PROMPT


# Local Build
def build_agent(mini: ReachyMini):
    return create_agent(
        model=Model(model="gemma4:12b", base_url="http://localhost:11434"),
        tools=build_toolbox(mini),
        system_prompt=SYSTEM_PROMPT,
    )
