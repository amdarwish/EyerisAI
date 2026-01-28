
# --- QA PROMPT for quality assurance use case ---
QA_PROMPT = (
    """
    You will receive input containing a 'description' field with frame analysis data in JSON format. Each key represents a frame index and the value describes that frame.
    
    Analyze the frame descriptions and determine if there are any issues mentioned.
    
    Return EXACTLY ONE JSON object with this format:
    - If ANY frame indicates an issue: {"alert": true, "tool_call": {"name": "send_email_alert_tool"}, "reason": "brief explanation of what issues were found"}
    - If NO issues are found: {"alert": false, "reason": "brief explanation why no alert is needed"}
    
    Look for keywords indicating problems like "issue", "problem", "piling up high", "can fall", etc.
    """
)

# --- Add this new QA node ---
def qa_model_node(state):
    try:
        messages = state["messages"]
        latest_human_message = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage) or (isinstance(msg, dict) and msg.get('role') == 'user')), None)
        if latest_human_message is None:
            raise RuntimeError('No human message provided to agent')
        human_content = latest_human_message.get('content') if isinstance(latest_human_message, dict) else latest_human_message.content

        api_conf = CONFIG['ai']
        model = api_conf.get('agent_model') or api_conf.get('model')
        headers = {'Content-Type': 'application/json'}
        if api_conf.get('api_key'):
            headers['Authorization'] = f"Bearer {api_conf.get('api_key')}"

        prompt = QA_PROMPT + "\nInput description:\n" + human_content
        payload = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': 'You are a strict format-enforcing assistant. Output exactly one JSON object and nothing else.'},
                {'role': 'user', 'content': prompt}
            ],
            'max_tokens': 1000,
            # 'temperature': 0.0,
            # 'top_p': 1.0,
            'seed': 42
        }
        base_url = api_conf.get('base_url')
        resp = requests.post(f"{base_url}/v1/chat/completions", headers=headers, json=payload, timeout=360)
        if resp.status_code == 200:
            out = resp.json()['choices'][0]['message']['content']
            return {"messages": [{"role": "user", "content": out}]}
        else:
            raise RuntimeError(f'LLM request failed: {resp.status_code} {resp.text}')
    except Exception as e:
        return {"messages": [{"role": "assistant", "content": f"Error: {e}"}]}

# --- QA tool dispatch node (reuses send_email_alert_tool) ---
def qa_tool_dispatch_node(state):
    try:
        messages = state["messages"]
        last_message = messages[-1]
        tool_calls = []
        content = last_message.get('content') if isinstance(last_message, dict) else getattr(last_message, 'content', None)
        if not isinstance(content, str):
            content = ''
        import re
        try:
            m = re.search(r"\{[\s\S]*\}", content)
            if m:
                data = json.loads(m.group(0))
                if isinstance(data, dict) and data.get('tool_call') and isinstance(data['tool_call'], dict):
                    tc = data['tool_call']
                    args_obj = tc.get('args', {})
                    tool_calls.append({
                        'name': tc.get('name'),
                        'args': json.dumps(args_obj),
                        'id': 'tool_call_json_qa_0'
                    })
                elif isinstance(data, dict) and data.get('alert'):
                    if data.get('tool_call') and isinstance(data['tool_call'], dict):
                        tc = data['tool_call']
                        args_obj = tc.get('args', {})
                        tool_calls.append({
                            'name': tc.get('name'),
                            'args': json.dumps(args_obj),
                            'id': 'tool_call_json_qa_1'
                        })
                    else:
                        args_obj = {}
                        for k in ('image_path', 'description', 'timestamp', 'frames_b64'):
                            if k in data:
                                args_obj[k] = data[k]
                        if args_obj:
                            tool_calls.append({
                                'name': 'send_email_alert_tool',
                                'args': json.dumps(args_obj),
                                'id': 'tool_call_json_qa_2'
                            })
        except Exception as e:
            print(f"qa_tool_dispatch_node: JSON parse attempt failed: {e}")
        matches = re.findall(r'<tool_call name="([^"]+)" args="([\s\S]*?)"\s*/>', content)
        for idx, (tool_name, args) in enumerate(matches):
            tool_calls.append({
                'name': tool_name,
                'args': args,
                'id': f'tool_call_xml_qa_{idx}'
            })
        new_messages = []
        for call in tool_calls:
            tool_name = call["name"]
            args = call["args"]
            print(f"[QA] Registering tool call: {tool_name} with args (raw): {args[:200] if args else args}")
            if tool_name == 'send_email_alert_tool':
                new_messages.append(
                    ToolMessage(content=str(args), name=tool_name, tool_call_id=call.get('id'))
                )
                print(f"[QA] Tool {tool_name} registered (execution deferred to caller).")
                continue
        print(f"[QA] Tool dispatch node returning {len(new_messages)} messages.")
        return {"messages": new_messages}
    except Exception as e:
        print(f"Error in qa_tool_dispatch_node: {e}")
        error_msg = ToolMessage(
            content="An error occurred while dispatching QA tools.",
            name="qa_tool_dispatch_node",
            tool_call_id=None
        )
        return {"messages": [error_msg]}


# For QA, you can use a separate entry or conditional path. Here, we provide a public helper:

def run_qa_agent(description: str, frames_bytes: list, image_path: str, timestamp: str):
    try:
        frames_trimmed = frames_bytes[:6]
        frames_b64 = [base64.b64encode(b).decode('utf-8') for b in frames_trimmed]
        global LAST_EVENT_CONTEXT
        LAST_EVENT_CONTEXT = {
            'description': description,
            'frames_bytes': frames_trimmed,
            'frames_b64': frames_b64,
            'image_path': image_path,
            'timestamp': timestamp
        }
        payload = {
            'description': description
        }
        print('[qa_agent] Invoking QA graph with payload')
        # Run only the QA nodes
        qa_graph = StateGraph(MessagesState)
        qa_graph.add_node("qa_model_node", qa_model_node)
        qa_graph.add_node("qa_tools", qa_tool_dispatch_node)
        qa_graph.add_edge(START, "qa_model_node")
        qa_graph.add_conditional_edges("qa_model_node", should_continue, ["qa_tools", END])
        qa_graph.add_edge("qa_tools", "qa_model_node")
        qa_graph_compiled = qa_graph.compile()
        result = qa_graph_compiled.invoke({
            'messages': [
                {'role': 'user', 'content': json.dumps(payload)}
            ]
        })
        print(f"[qa_agent] QA graph invoke returned: {str(result)[:1000]}")
        text = ''
        if isinstance(result, dict) and 'messages' in result and result['messages']:
            # Find the last non-empty message content
            for msg in reversed(result['messages']):
                content = msg.get('content') if isinstance(msg, dict) else getattr(msg, 'content', '')
                if content and content.strip():
                    text = content
                    break
        else:
            text = str(result)
        import re
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            print('[qa_agent] No JSON found in model output; not sending QA email.')
            return False
        try:
            data = json.loads(m.group(0))
        except Exception as e:
            print(f"Failed to parse JSON from QA model output: {e}")
            return False
        reason = data.get('reason')
        tool_call = data.get('tool_call')
        if tool_call and isinstance(tool_call, dict) and tool_call.get('name') == 'send_email_alert_tool':
            if data.get('alert'):
                print('[qa_agent] QA agent requested send_email_alert_tool and alert:true; sending QA email with full context')
                send_email_alert_tool(
                    image_path=LAST_EVENT_CONTEXT.get('image_path'),
                    description=LAST_EVENT_CONTEXT.get('description'),
                    timestamp=LAST_EVENT_CONTEXT.get('timestamp'),
                    frames_bytes=LAST_EVENT_CONTEXT.get('frames_bytes'),
                    reason=reason
                )
                return True
            else:
                print('[qa_agent] QA agent included tool_call for send_email_alert_tool but alert:false -> not sending QA email. Reason: ' + (reason or 'no reason provided'))
                return False
        if data.get('alert'):
            print('[qa_agent] QA agent returned alert:true without explicit tool_call; sending QA email')
            send_email_alert_tool(
                image_path=LAST_EVENT_CONTEXT.get('image_path'),
                description=LAST_EVENT_CONTEXT.get('description'),
                timestamp=LAST_EVENT_CONTEXT.get('timestamp'),
                frames_bytes=LAST_EVENT_CONTEXT.get('frames_bytes'),
                reason=reason
            )
            return True
        print('[qa_agent] QA agent decided NOT to send QA email: ' + (reason or 'no reason provided'))
        return False
    except Exception as e:
        print(f"run_qa_agent error: {e}")
        print('[qa_agent] QA agent error fallback: not sending QA email')
        return False
from langgraph.graph import END
from datetime import datetime
import pytz
from langchain_core.messages import ToolMessage, HumanMessage, AnyMessage
from typing import Annotated
from langgraph.graph.message import add_messages
from typing import TypedDict
from langgraph.graph import StateGraph, MessagesState, START
import base64
import configparser
import json
import os
import smtplib
import time
from datetime import datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
import logging
import requests

# Local lightweight config loader to avoid circular imports with EyerisAI
def load_config_local(config_path: str | os.PathLike | None = None):
    config = configparser.ConfigParser()
    if config_path is None:
        # Default to config.ini next to this file, so imports work regardless of CWD.
        config_path = Path(__file__).with_name("config.ini")
    config.read(str(config_path))
    return {
        'instance_name': config.get('General', 'instance_name', fallback='Motion Detector'),
        'ai': {
            'base_url': config.get('AI', 'base_url', fallback=''),
            'model': config.get('AI', 'model', fallback='gpt-4o'),
            'agent_model': config.get('AI', 'agent_model', fallback=config.get('AI', 'model', fallback='gpt-4o')),
            'api_key': config.get('AI', 'api_key', fallback=None),
        },
        'email': {
            'enabled': config.getboolean('Email', 'enabled', fallback=False),
            'smtp_server': config.get('Email', 'smtp_server', fallback=''),
            'smtp_port': config.getint('Email', 'smtp_port', fallback=25),
            'smtp_username': config.get('Email', 'smtp_username', fallback=''),
            'smtp_password': config.get('Email', 'smtp_password', fallback=''),
            'from_address': config.get('Email', 'from_address', fallback=''),
            'to_address': config.get('Email', 'to_address', fallback=''),
            'use_tls': config.getboolean('Email', 'use_tls', fallback=True)
        }
    }

# Load configuration at module import time using the local loader
CONFIG = load_config_local()

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def should_continue(state):
    try:
        # Always end after the model node; run_model will produce the final decision JSON.
        return END
    except Exception as e:
        print(f"Error in should_continue: {e}")
        return END


def call_model(state):
    try:
        messages = state["messages"]
        print(f"Received {len(messages)} messages.")

        # Find the latest HumanMessage
        latest_human_message = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage) or (isinstance(msg, dict) and msg.get('role') == 'user')), None)
        # Normalize to a content string
        if latest_human_message is None:
            raise RuntimeError('No human message provided to agent')

        if isinstance(latest_human_message, dict):
            human_content = latest_human_message.get('content')
        else:
            human_content = latest_human_message.content

        print('[agent] Calling LLM to decide whether to trigger tools')

        api_conf = CONFIG['ai']
        model = api_conf.get('agent_model') or api_conf.get('model')
        headers = {'Content-Type': 'application/json'}
        if api_conf.get('api_key'):
            headers['Authorization'] = f"Bearer {api_conf.get('api_key')}"

        # Simplified, strict prompt: require exactly one JSON object as output and nothing else.
        prompt = (
            "A single JSON object decision is required. Input below is the description whether human/person is detected in the frames.\n"
            "Decision rule: if the description indicates a person or human is present in the frames (including stationary or still people), set \"alert\": true . If alert is true include a \"tool_call\" object with \"name\": \"send_email_alert_tool\". Always include a brief \"reason\" explaining the decision.\n"
            "Return EXACTLY ONE JSON OBJECT and NOTHING ELSE.\n\n"
            "Input description:\n"
            f"{human_content}\n"
        )

        payload = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': 'You are a strict format-enforcing assistant. Output exactly one JSON object and nothing else.'},
                {'role': 'user', 'content': prompt}
            ],
            'max_tokens': 1000,
            # 'temperature': 0.0,
            # 'top_p': 1.0,
            'seed': 42
        }

        base_url = api_conf.get('base_url')
        if not base_url:
            raise RuntimeError('AI base_url not configured')

        resp = requests.post(f"{base_url}/v1/chat/completions", headers=headers, json=payload, timeout=360)
        if resp.status_code == 200:
            try:
                out = resp.json()['choices'][0]['message']['content']
                print('[agent] LLM call success')
                # Return only the model content as the final message; do NOT append markers or request tool execution here
                return {"messages": [{"role": "user", "content": out}]}
            except Exception as e:
                print(f'[agent] Failed to parse LLM response JSON: {e}; returning raw text')
                return {"messages": [{"role": "user", "content": resp.text}]}
        else:
            msg = f'LLM request failed: {resp.status_code} {resp.text}'
            print(f'[agent] {msg}')
            raise RuntimeError(msg)

    except Exception as e:
        print(f"Error in call_model: {e}")
        error_message = {
            "role": "assistant",
            "content": "Sorry, an error occurred while processing your request. Please try again later."
        }
        return {"messages": [error_message]}


def tool_dispatch_node(state):
    try:
        messages = state["messages"]
        last_message = messages[-1]
        # Parse tool call from message content
        tool_calls = []
        content = None
        if isinstance(last_message, dict):
            content = last_message.get('content')
        else:
            content = getattr(last_message, 'content', None)

        if not isinstance(content, str):
            content = ''

        import re

        # First, try extracting JSON object and look for 'tool_call' structure
        try:
            m = re.search(r"\{[\s\S]*\}", content)
            if m:
                data = json.loads(m.group(0))
                # If model provided the tool_call object, convert to our internal call format
                if isinstance(data, dict) and data.get('tool_call') and isinstance(data['tool_call'], dict):
                    tc = data['tool_call']
                    args_obj = tc.get('args', {})
                    tool_calls.append({
                        'name': tc.get('name'),
                        'args': json.dumps(args_obj),
                        'id': 'tool_call_json_0'
                    })
                # Legacy: if the model simply returned alert:true and included args at top-level
                elif isinstance(data, dict) and data.get('alert'):
                    # try to find send args under 'tool_call' or top-level arg keys
                    if data.get('tool_call') and isinstance(data['tool_call'], dict):
                        tc = data['tool_call']
                        args_obj = tc.get('args', {})
                        tool_calls.append({
                            'name': tc.get('name'),
                            'args': json.dumps(args_obj),
                            'id': 'tool_call_json_1'
                        })
                    else:
                        # build args from expected top-level keys
                        args_obj = {}
                        for k in ('image_path', 'description', 'timestamp', 'frames_b64'):
                            if k in data:
                                args_obj[k] = data[k]
                        if args_obj:
                            tool_calls.append({
                                'name': 'send_email_alert_tool',
                                'args': json.dumps(args_obj),
                                'id': 'tool_call_json_2'
                            })
        except Exception as e:
            # JSON extraction failed; continue to XML fallback
            print(f"tool_dispatch_node: JSON parse attempt failed: {e}")

        # Fallback: match XML-style tag if present
        matches = re.findall(r'<tool_call name="([^\"]+)" args="([\s\S]*?)"\s*/>', content)
        for idx, (tool_name, args) in enumerate(matches):
            tool_calls.append({
                'name': tool_name,
                'args': args,
                'id': f'tool_call_xml_{idx}'
            })

        new_messages = []
        tool_map = {
            "send_email_alert_tool": send_email_alert_tool
        }

        # Default: register tool calls. For send_email_alert_tool we DO NOT execute it here; we only register the call so the
        # caller (run_agent) can execute it using the full event context (frames, path, timestamp).
        for call in tool_calls:
            tool_name = call["name"]
            args = call["args"]
            print(f"Registering tool call: {tool_name} with args (raw): {args[:200] if args else args}")
            # For email alerts, do not execute in the graph; return a ToolMessage describing the call
            if tool_name == 'send_email_alert_tool':
                # Add a ToolMessage to indicate the tool was requested
                new_messages.append(
                    ToolMessage(content=str(args), name=tool_name, tool_call_id=call.get('id'))
                )
                print(f"Tool {tool_name} registered (execution deferred to caller).")
                continue

            # For other tools, attempt to execute as before
            tool_func = tool_map.get(tool_name)
            if not tool_func:
                print(f"Tool {tool_name} not found in tool_map.")
                continue
            try:
                parsed = None
                if isinstance(args, str):
                    try:
                        parsed = json.loads(args)
                    except Exception:
                        try:
                            unescaped = args.encode('utf-8').decode('unicode_escape')
                            parsed = json.loads(unescaped)
                        except Exception:
                            parsed = None
                if isinstance(parsed, dict):
                    result = tool_func(**parsed)
                else:
                    result = tool_func(args)

                new_messages.append(
                    ToolMessage(content=str(result), name=tool_name, tool_call_id=call["id"])
                )
                print(f"Tool {tool_name} executed successfully.")
            except Exception as tool_error:
                print(f"Error executing tool {tool_name}: {tool_error}")
                error_msg = ToolMessage(
                    content=f"Error executing tool {tool_name}: {tool_error}",
                    name=tool_name,
                    tool_call_id=call["id"]
                )
                new_messages.append(error_msg)
        print(f"Tool dispatch node returning {len(new_messages)} messages.")
        return {"messages": new_messages}

    except Exception as e:
        print(f"Error in tool_dispatch_node: {e}")
        error_msg = ToolMessage(
            content="An error occurred while dispatching tools.",
            name="tool_dispatch_node",
            tool_call_id=None
        )
        return {"messages": [error_msg]}



# Public helper to run the agent for a motion event. This will invoke the graph and
# prefer the agent's decision if available; otherwise fall back to a simple keyword check.
# Module-level context for the last event so tools can access full data even when LLM only sees description
LAST_EVENT_CONTEXT = {}


def run_motion_agent(description: str, frames_bytes: list, image_path: str, timestamp: str):
    """
    Run the motion/human detection agent. This builds and uses the motion_graph (call_model/tools).
    """
    try:
        # Limit frames to at most 6 for sending
        frames_trimmed = frames_bytes[:6]
        frames_b64 = [base64.b64encode(b).decode('utf-8') for b in frames_trimmed]

        # Store full event context in module-level variable for deferred tool execution
        global LAST_EVENT_CONTEXT
        LAST_EVENT_CONTEXT = {
            'description': description,
            'frames_bytes': frames_trimmed,
            'frames_b64': frames_b64,
            'image_path': image_path,
            'timestamp': timestamp
        }

        payload = {
            'description': description
        }

        # Build the motion graph (call_model/tools)
        builder = StateGraph(MessagesState)
        builder.add_node("call_model", call_model)
        builder.add_node("tools", tool_dispatch_node)
        builder.add_edge(START, "call_model")
        builder.add_conditional_edges("call_model", should_continue, ["tools", END])
        builder.add_edge("tools", "call_model")
        motion_graph = builder.compile()

        print('[motion_agent] Invoking motion_graph with payload')
        result = motion_graph.invoke({
            'messages': [
                {'role': 'user', 'content': json.dumps(payload)}
            ]
        })

        print(f"[motion_agent] Graph invoke returned: {str(result)[:1000]}")

        # Extract model output (final JSON) from result
        text = ''
        if isinstance(result, dict) and 'messages' in result and result['messages']:
            m = result['messages'][-1]
            text = m.get('content') if isinstance(m, dict) else getattr(m, 'content', str(m))
        else:
            text = str(result)

        # Try to extract first JSON object
        import re
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            print('[motion_agent] No JSON found in model output; falling back to heuristic')
            # Fallback heuristic
            if 'person' in description.lower() or 'human' in description.lower():
                reason = 'heuristic: description contains person/human'
                send_email_alert_tool(image_path=image_path, description=description, timestamp=timestamp, frames_bytes=frames_trimmed, reason=reason)
                return True
            else:
                print('[motion_agent] Fallback heuristic: no person detected -> not sending email')
                return False

        try:
            data = json.loads(m.group(0))
        except Exception as e:
            print(f"Failed to parse JSON from model output: {e}")
            return False

        # Expect data to contain 'alert' and optional 'tool_call' and 'reason'
        reason = data.get('reason')
        tool_call = data.get('tool_call')

        # If model provided a tool_call, only honor it when alert is explicitly true.
        if tool_call and isinstance(tool_call, dict) and tool_call.get('name') == 'send_email_alert_tool':
            if data.get('alert'):
                print('[motion_agent] Agent requested send_email_alert_tool and alert:true; sending email with full context')
                send_email_alert_tool(
                    image_path=LAST_EVENT_CONTEXT.get('image_path'),
                    description=LAST_EVENT_CONTEXT.get('description'),
                    timestamp=LAST_EVENT_CONTEXT.get('timestamp'),
                    frames_bytes=LAST_EVENT_CONTEXT.get('frames_bytes'),
                    reason=reason
                )
                return True
            else:
                print('[motion_agent] Agent included tool_call for send_email_alert_tool but alert:false -> not sending email. Reason: ' + (reason or 'no reason provided'))
                return False

        # If explicit alert true without tool_call, honor it
        if data.get('alert'):
            print('[motion_agent] Agent returned alert:true without explicit tool_call; sending email')
            send_email_alert_tool(
                image_path=LAST_EVENT_CONTEXT.get('image_path'),
                description=LAST_EVENT_CONTEXT.get('description'),
                timestamp=LAST_EVENT_CONTEXT.get('timestamp'),
                frames_bytes=LAST_EVENT_CONTEXT.get('frames_bytes'),
                reason=reason
            )
            return True

        print('[motion_agent] Agent decided NOT to send email: ' + (reason or 'no reason provided'))
        return False

    except Exception as e:
        print(f"run_motion_agent error: {e}")
        # On error, fallback heuristic
        if 'person' in description.lower() or 'human' in description.lower():
            reason = 'heuristic (error fallback): description contains person/human'
            send_email_alert_tool(image_path=image_path, description=description, timestamp=timestamp, frames_bytes=frames_bytes[:6], reason=reason)
            return True
        else:
            print('[motion_agent] Agent error fallback: not sending email')
            return False


# Updated send_email_alert_tool to accept either kwargs or a single raw arg string
def send_email_alert_tool(image_path=None, description=None, timestamp=None, frames_bytes=None, reason=None, *args, **kwargs):
    """
    Send email alert with image and description.
    Accepts either explicit named args or a single JSON string/dict.
    """
    # If called with a single positional arg which is a JSON string, parse it
    if not description and args:
        try:
            parsed = json.loads(args[0])
            image_path = parsed.get('image_path')
            description = parsed.get('description')
            timestamp = parsed.get('timestamp')
            reason = reason or parsed.get('reason')
            frames_b64 = parsed.get('frames_b64')
            if frames_b64:
                frames_bytes = [base64.b64decode(x) for x in frames_b64]
        except Exception:
            # try kwargs fallback
            pass

    # If frames were provided as base64 strings in kwargs
    if frames_bytes is None and kwargs.get('frames_b64'):
        try:
            frames_bytes = [base64.b64decode(x) for x in kwargs.get('frames_b64')]
        except Exception:
            frames_bytes = None

    # Accept reason from kwargs too
    if not reason and 'reason' in kwargs:
        reason = kwargs.get('reason')

    if not CONFIG['email']['enabled']:
        print('[agent] Email sending is disabled in config; would have sent alert.')
        return

    email_config = CONFIG['email']

    # Create the email message
    msg = MIMEMultipart()
    subject_reason = (reason[:80] + '...') if reason and len(reason) > 80 else (reason or '')
    msg['Subject'] = f"{CONFIG['instance_name']} - Event Detected at {timestamp}" + (f" - {subject_reason}" if subject_reason else "")
    msg['From'] = email_config['from_address']
    msg['To'] = email_config['to_address']

    # Add description and agent reason as text
    agent_reason_text = reason if reason else 'No agent reason provided.'
    text = MIMEText(f"Motion Detection Alert\n\nTime: {timestamp}\n\nDescription: {description}\n\nAgent decision reason: {agent_reason_text}")
    msg.attach(text)

    # Attach frames
    if frames_bytes:
        for idx, img_bytes in enumerate(frames_bytes, start=1):
            try:
                img = MIMEImage(img_bytes)
                filename = f"{Path(image_path).stem}_frame{idx}.jpg" if image_path else f"frame{idx}.jpg"
                img.add_header('Content-Disposition', 'attachment', filename=filename)
                msg.attach(img)
            except Exception as e:
                print(f"Failed to attach frame {idx}: {e}")
    else:
        # Backwards compatible: attach the single saved image file
        try:
            if image_path and os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    img = MIMEImage(f.read())
                    img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
                    msg.attach(img)
        except Exception as e:
            print(f"Failed to attach image file '{image_path}': {e}")

    # Send the email
    try:
        with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
            if email_config['use_tls']:
                server.starttls()

            if email_config['smtp_username'] and email_config['smtp_password']:
                server.login(email_config['smtp_username'], email_config['smtp_password'])

            server.send_message(msg)
            print(f"Email alert sent to {email_config['to_address']} (reason: {agent_reason_text})")
    except Exception as e:
        print(f"Failed to send email alert: {str(e)}")


if __name__ == "__main__":
    try:
        print("Starting graph invocation with test message.")
        result = graph.invoke({
            "messages": [
                {"role": "user", "content": "book a meeting tommorow 9 am"}
            ]
        })
        print(f"Graph invocation result: {result}")
        print(result)
    except Exception as e:
        print(f"Error during graph invocation: {e}")
        print(f"Error during graph invocation: {e}")
