import queue
import threading
import time

from reachy_mini import ReachyMini

from agent_reachy import stt, tts
from agent_reachy.agent import build_agent
from agent_reachy.tools import ToolLoggingHandler

ITALIC = "\033[3m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def _read_stdin(input_queue: "queue.Queue", input_ready: threading.Event) -> None:
    while True:
        input_ready.wait()
        input_ready.clear()
        try:
            line = input("You: ")
        except EOFError:
            input_queue.put(("text", "exit"))
            return
        input_queue.put(("text", line))
        if line in ("exit", "quit"):
            return


def main() -> None:
    with ReachyMini() as mini:
        agent = build_agent(mini)

        time.sleep(2)  # Wait for robot and agent to be ready

        input_queue = queue.Queue()
        stop_event = threading.Event()
        speaking_event = threading.Event()
        input_ready = threading.Event()
        input_ready.set()

        threading.Thread(target=_read_stdin, args=(input_queue, input_ready), daemon=True).start()
        stt_thread = threading.Thread(
            target=stt.listen_loop,
            args=(mini, lambda text: input_queue.put(("voice", text)), stop_event, speaking_event),
            daemon=True,
        )
        stt_thread.start()

        messages = []
        try:
            while True:
                source, user_input = input_queue.get()
                if source == "voice":
                    print(f"You (voice): {user_input}")
                if user_input in ("exit", "quit"):
                    break

                messages.append({"role": "user", "content": user_input})

                current_id = None
                thinking_started = False
                answer_started = False
                for mode, data in agent.stream(
                    {"messages": messages},
                    config={"callbacks": [ToolLoggingHandler()]},
                    stream_mode=["messages", "values"],
                ):
                    if mode == "values":
                        messages = data["messages"]
                        continue

                    chunk, _metadata = data
                    if chunk.id != current_id:
                        if thinking_started:
                            print(RESET, end="")
                        if answer_started or thinking_started:
                            print()
                        current_id = chunk.id
                        thinking_started = False
                        answer_started = False

                    reasoning = chunk.additional_kwargs.get("reasoning_content")
                    if reasoning:
                        if not thinking_started:
                            print(f"{ITALIC}{YELLOW}Thinking:", end=" ", flush=True)
                            thinking_started = True
                        print(reasoning, end="", flush=True)
                    if chunk.content:
                        if not answer_started:
                            if thinking_started:
                                print(RESET)
                            print("Agent:", end=" ", flush=True)
                            answer_started = True
                        print(chunk.content, end="", flush=True)
                if thinking_started:
                    print(RESET, end="")
                if answer_started or thinking_started:
                    print()

                if messages:
                    speaking_event.set()
                    try:
                        tts.speak(mini, messages[-1].content)
                    finally:
                        speaking_event.clear()

                input_ready.set()
        finally:
            stop_event.set()
            stt_thread.join(timeout=2)


if __name__ == "__main__":
    main()
