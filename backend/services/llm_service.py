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
        Takes the state dictionary and returns an action index (0-10).
        """
        revenue = state_data.get("revenue", 0)
        cash = state_data.get("cash_balance", 0)
        morale = state_data.get("employee_morale", 0)
        
        prompt = f"""You are the CPU/CEO of ATLAS.
Company Metrics:
- Revenue: ${revenue}
- Cash: ${cash}
- Employee Morale: {morale}/10

Your available actions:
0 - Relax
1 - Start New Feature
2 - Review Code
3 - Outbound Campaign
4 - Customer Followups
5 - Post Job Ad
6 - Conduct Interviews
7 - Team Building Event
8 - Review Financials
9 - Cut Costs
10 - Customer Support

Reply with ONLY a single number from 0 to 10."""

        if self.gemini_key:
            return self._call_gemini(prompt)
        elif self.openai_key:
            return self._call_openai(prompt)
        elif self.anthropic_key:
            return self._call_anthropic(prompt)
        elif self.hf_token:
            return self._call_huggingface(prompt)
        else:
            return random.randint(0, 10)

    def _call_gemini(self, prompt):
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            prediction = response.text.strip()
            digits = "".join([c for c in prediction if c.isdigit()])
            if digits:
                return int(digits)
        except Exception as e:
            logger.error(f"Gemini API Error: {e}")
        return random.randint(0, 10)

    def _call_anthropic(self, prompt):
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=self.anthropic_key)
            message = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}]
            )
            prediction = message.content[0].text.strip()
            digits = "".join([c for c in prediction if c.isdigit()])
            if digits:
                return int(digits)
        except Exception as e:
            logger.error(f"Anthropic API Error: {e}")
        return random.randint(0, 10)

    def _call_huggingface(self, prompt):
        API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        hf_prompt = f"<s>[INST] {prompt} [/INST]"
        payload = {
            "inputs": hf_prompt,
            "parameters": {"max_new_tokens": 5, "temperature": 0.1, "return_full_text": False}
        }
        
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            prediction = response.json()[0]["generated_text"].strip()
            digits = "".join([c for c in prediction if c.isdigit()])
            if digits:
                return int(digits)
        except Exception as e:
            logger.error(f"HuggingFace API Error: {e}")
        return random.randint(0, 10)

    def _call_openai(self, prompt):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.5
            )
            prediction = response.choices[0].message.content.strip()
            digits = "".join([c for c in prediction if c.isdigit()])
            if digits:
                return int(digits)
        except Exception as e:
            logger.error(f"OpenAI API Error: {e}")
        return random.randint(0, 10)
