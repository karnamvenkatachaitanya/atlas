import os
import random
import requests
import logging

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.hf_token = os.getenv("HF_TOKEN")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        
    def is_enabled(self):
        return bool(self.hf_token or self.openai_key or self.gemini_key or self.anthropic_key)

    def get_action(self, state_data):
        """
        Takes the state dictionary and returns an action dict:
        {"action": int, "intensity": float, "rationale": str}
        """
        import re
        revenue = state_data.get("revenue", 0)
        cash = state_data.get("cash_balance", 0)
        morale = state_data.get("employee_morale", 0)
        trust = state_data.get("investor_trust", 0)
        mandate = state_data.get("mandate", "Balanced Stability")
        inbox = state_data.get("inbox", "No messages.")
        
        prompt = f"""You are the CEO of ATLAS. 
Your goal is to manage a startup over 90 days.
Current Board Mandate: {mandate}

Company Metrics:
- Revenue: ${revenue}
- Cash: ${cash}
- Employee Morale: {morale}/100
- Investor Trust: {trust}/100

Department Inbox:
{inbox}

Your available actions (Reply with ONLY the index number):
0 - Hire Employee (Increases burn, increases progress)
1 - Fire Employee (Decreases burn, decreases morale)
2 - Increase Salaries (Increases morale, increases burn)
3 - Assign Engineering Task (Increases product progress)
4 - Launch Product (Increases revenue, uses progress)
5 - Run Ads (Increases revenue, increases burn)
6 - Negotiate Client (Increases revenue, increases trust)
7 - Reduce Costs (Decreases burn, decreases morale)
8 - Raise Funding (Increases cash, decreases trust)
9 - Fix Bug/Crisis (Increases satisfaction, fixes crises)
10 - Improve Culture (Increases morale, increases burn)
11 - Give Bonuses (Decreases cash, increases morale)
12 - Change Roadmap (Increases progress, increases tasks)

You must explain your thought process and pick an intensity from 0.1 to 1.0.
Reply STRICTLY using this format:
<rationale>Your detailed reasoning here based on the mandate and metrics.</rationale>
<action>The single integer index of your action</action>
<intensity>A float between 0.1 and 1.0</intensity>
"""

        prediction = ""
        if self.gemini_key:
            prediction = self._call_gemini(prompt)
        elif self.openai_key:
            prediction = self._call_openai(prompt)
        elif self.anthropic_key:
            prediction = self._call_anthropic(prompt)
        elif self.hf_token:
            prediction = self._call_huggingface(prompt)
            
        return self._parse_llm_output(prediction)
        
    def _parse_llm_output(self, prediction: str) -> dict:
        import re
        action_idx = random.randint(0, 12)
        intensity = 1.0
        rationale = "No rationale provided."
        
        if not prediction:
            return {"action": action_idx, "intensity": intensity, "rationale": "Fallback random action."}
            
        rat_match = re.search(r"<rationale>(.*?)</rationale>", prediction, re.DOTALL)
        act_match = re.search(r"<action>(.*?)</action>", prediction, re.DOTALL)
        int_match = re.search(r"<intensity>(.*?)</intensity>", prediction, re.DOTALL)
        
        if rat_match:
            rationale = rat_match.group(1).strip()
        if act_match:
            digits = "".join([c for c in act_match.group(1) if c.isdigit()])
            if digits:
                action_idx = int(digits)
        if int_match:
            try:
                intensity = float(int_match.group(1).strip())
                intensity = max(0.1, min(1.0, intensity))
            except ValueError:
                pass
                
        return {"action": action_idx, "intensity": intensity, "rationale": rationale}

    def _call_gemini(self, prompt):
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini API Error: {e}")
        return ""

    def _call_anthropic(self, prompt):
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=self.anthropic_key)
            message = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text.strip()
        except Exception as e:
            logger.error(f"Anthropic API Error: {e}")
        return ""

    def _call_huggingface(self, prompt):
        API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        hf_prompt = f"<s>[INST] {prompt} [/INST]"
        payload = {
            "inputs": hf_prompt,
            "parameters": {"max_new_tokens": 300, "temperature": 0.1, "return_full_text": False}
        }
        
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            return response.json()[0]["generated_text"].strip()
        except Exception as e:
            logger.error(f"HuggingFace API Error: {e}")
        return ""

    def _call_openai(self, prompt):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.5
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API Error: {e}")
        return ""
