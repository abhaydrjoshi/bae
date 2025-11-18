#!/usr/bin/env python3
"""
bae.py - Minimal conversational agent with a simple Torch scorer.
Focus: simplicity and compatibility with Trainer.
"""

import re
import random
from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn

from models import ScoringMLP

CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "n't": " not",
    "'re": " are", "'ve": " have", "'ll": " will",
    "'d": " would", "'m": " am",
}

RULES = [
    (r"\bi\s+love\s+(.*)", [
        "Why do you love %1?",
        "What about %1 makes you feel that way?",
        "Tell me more about your feelings towards %1."
    ]),
    (r"\bi\s+hate\s+(.*)", [
        "What makes you hate %1?",
        "Can you tell me more about that feeling?",
        "Has %1 always made you feel this way?"
    ]),
    (r"\bwhy\s+(.*)", [
        "That's a great question! Why do you think %1?",
        "What leads you to ask about %1?",
        "What answers have you considered?"
    ]),
    (r".*", [
        "I hear you. Tell me more.",
        "What feels most important about that?",
        "That's interesting. Could you elaborate?",
        "What else comes to mind?"
    ]),
]


def normalize(text: str) -> Tuple[str, List[str]]:
    if not isinstance(text, str) or not text:
        return "", []
    text = text.strip().lower()
    for a, b in sorted(CONTRACTIONS.items(), key=lambda x: len(x[0]), reverse=True):
        text = re.sub(re.escape(a), b, text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = re.findall(r"\b\w+\b", text)
    return text, words


class Bae:
    def __init__(self):
        self.memory: List[str] = []
        self.metrics: Dict[str, int] = {"inputs": 0, "matches": 0, "stored_memories": 0}
        self.compiled_rules: List[Tuple[re.Pattern, List[str]]] = [
            (re.compile(p, re.I), r) for p, r in RULES
        ]
        # ~100k-parameter MLP scorer over 4 engineered features
        self.scorer = ScoringMLP(input_dim=4, output_dim=1)
        # Initialize final layer bias to produce scores in [0,1] after clamping
        for m in self.scorer.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self._word_swaps = {
            "i": "you", "me": "you", "my": "your", "mine": "yours",
            "you": "I", "your": "my", "am": "are", "are": "am"
        }

    def _swap(self, phrase: str) -> str:
        if not phrase:
            return phrase
        words = phrase.lower().split()
        swapped = [self._word_swaps.get(w, w) for w in words]
        result = []
        for i, word in enumerate(swapped):
            if word == "i" or i == 0:
                result.append(word.capitalize())
            else:
                result.append(word)
        return " ".join(result)

    def _score_response(self, response: str, input_words: List[str]) -> float:
        response_lower = response.lower()
        keyword_match = any(w in response_lower for w in input_words) if input_words else False
        relevance = float(keyword_match)
        response_words = response.split()
        length = min(len(response_words) / 10.0, 1.0)
        empathy_words = ['feel', 'understand', 'hear', 'seems', 'sense', 'notice']
        empathy = sum(w in response_lower for w in empathy_words) / len(empathy_words)
        diversity = len(set(response_words)) / max(len(response_words), 1)
        diversity = 1.0 if diversity > 0.6 else 0.5
        feats = torch.tensor([[empathy, relevance, length, diversity]], dtype=torch.float32)
        with torch.no_grad():
            score = self.scorer(feats).item()
        return max(0.0, min(score, 1.0))

    def respond(self, text: str, explain: bool = False) -> str:
        self.metrics["inputs"] += 1
        if not isinstance(text, str) or not text:
            return "Please say something meaningful."
        if len(text) > 2000:
            return "Please keep your input shorter."

        norm_text, words = normalize(text)
        if not norm_text:
            return "Please say something meaningful."

        for pattern, responses in self.compiled_rules:
            m = pattern.search(norm_text)
            if not m:
                continue
            self.metrics["matches"] += 1
            tmpl = random.choice(responses)
            reply = tmpl
            for i, g in enumerate(m.groups(), start=1):
                if g:
                    g_clean = g.strip()
                    g_swapped = self._swap(g_clean) if i == 1 else g_clean
                    reply = reply.replace(f"%{i}", g_swapped)
            if reply:
                reply = reply[0].upper() + reply[1:] if len(reply) > 1 else reply.upper()
            if explain:
                score = self._score_response(reply, words)
                return f"{reply} [score={score:.2f}]"
            return reply

        return random.choice(RULES[-1][1])


def bae() -> Bae:
    return Bae()



