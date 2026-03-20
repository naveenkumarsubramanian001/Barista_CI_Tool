import spacy
from transformers import pipeline
import json


class QueryAnalyzer:
    def __init__(self):
        # Load spaCy NER model
        self.nlp = spacy.load("en_core_web_sm")

        # Load zero-shot classifier for intent detection
        self.classifier = pipeline(
            "zero-shot-classification", model="facebook/bart-large-mnli"
        )

    def analyze(self, query):
        # Step 1: spaCy NER for entities
        doc = self.nlp(query)
        entities = [
            (ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents
        ]

        # Step 2: POS tags for structure analysis
        pos_tags = [(token.text, token.pos_, token.dep_) for token in doc]

        # Step 3: Intent classification
        intent_labels = [
            "recommendation",
            "comparison",
            "explanation",
            "search",
            "purchase",
            "analysis",
        ]
        intent_result = self.classifier(query, intent_labels)
        top_intent = intent_result["labels"][0]
        intent_score = intent_result["scores"][0]

        # Step 4: Extract constraints (numbers, dates, money)
        constraints = self._extract_constraints(doc)

        # Compile structured output
        analysis = {
            "original_query": query,
            "entities": entities,
            "pos_tags": pos_tags[:10],  # Top 10 for brevity
            "top_intent": top_intent,
            "intent_confidence": round(intent_score, 3),
            "constraints": constraints,
            "query_length": len(query.split()),
        }

        return analysis

    def _extract_constraints(self, doc):
        constraints = {}
        for ent in doc.ents:
            if ent.label_ in ["MONEY", "DATE", "CARDINAL", "PERCENT", "QUANTITY"]:
                constraints[ent.label_] = ent.text
        return constraints


# Usage
if __name__ == "__main__":
    analyzer = QueryAnalyzer()

    # Test queries
    test_queries = [
        "best budget AI laptops for ML in India under 1L INR",
        "compare PyTorch vs TensorFlow performance 2026",
        "explain Mamba architecture with code examples",
    ]

    for query in test_queries:
        result = analyzer.analyze(query)
        print(json.dumps(result, indent=2))
        print("-" * 80)
