#!/usr/bin/env python3
"""
Dataset Downloader for LLM Drift Analyzer

Downloads and converts benchmark datasets from HuggingFace to the project's
JSON format for both English and Hindi evaluation.

Datasets downloaded:
- English: IFEval (instruction following), TruthfulQA (factuality)
- Hindi: IndicQA (QA), Indic-Instruct samples

Usage:
    python scripts/download_datasets.py --language hi --output data/prompts/
    python scripts/download_datasets.py --language en --output data/prompts/
    python scripts/download_datasets.py --language all --output data/prompts/
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: 'datasets' library not installed. Run: pip install datasets")


def download_indicqa_hindi(max_samples: int = 30) -> List[Dict[str, Any]]:
    """
    Download Hindi QA dataset from ai4bharat/IndicMSMARCO.
    
    This is a passage retrieval dataset with queries and passages in Hindi.
    We convert it to factual_qa prompts.
    """
    print("Downloading ai4bharat IndicMSMARCO Hindi...")

    try:
        # Use streaming to avoid loading entire dataset
        dataset = load_dataset("ai4bharat/IndicMSMARCO", "hi", split="train", streaming=True)

        prompts = []
        seen_queries = set()

        for idx, item in enumerate(dataset):
            if len(prompts) >= max_samples:
                break

            query = item.get("query", "").strip()
            passage = item.get("passage", "").strip()
            answer = item.get("answer", "").strip()

            # Skip duplicates or empty queries
            if not query or query in seen_queries:
                continue
            seen_queries.add(query)

            # Create prompt - use query as question
            prompt_text = query
            
            # If passage is short, include it as context
            if passage and len(passage) < 300:
                prompt_text = f"संदर्भ: {passage[:200]}...\n\nप्रश्न: {query}"

            prompts.append({
                "id": f"HI-MSMARCO-{idx+1:03d}",
                "text": prompt_text,
                "category": "factual_qa",
                "language": "hi",
                "description": f"IndicMSMARCO Hindi - Question answering",
                "reference_answer": answer[:500] if answer else None,
                "source": "ai4bharat/IndicMSMARCO",
                "metadata": {
                    "difficulty": "medium",
                    "topic": "general_knowledge",
                    "dataset": "IndicMSMARCO"
                }
            })

        print(f"  Downloaded {len(prompts)} IndicMSMARCO Hindi prompts")
        return prompts

    except Exception as e:
        print(f"  Error downloading IndicMSMARCO: {e}")
        return []


def download_xnli_hindi(max_samples: int = 20) -> List[Dict[str, Any]]:
    """
    Download XNLI Hindi for reasoning/inference prompts.

    Convert premise-hypothesis pairs to reasoning questions.
    """
    print("Downloading XNLI Hindi...")

    try:
        dataset = load_dataset("facebook/xnli", "hi", split="test")

        prompts = []
        label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}

        for idx, item in enumerate(dataset):
            if len(prompts) >= max_samples:
                break

            premise = item.get("premise", "").strip()
            hypothesis = item.get("hypothesis", "").strip()
            label = item.get("label", 1)

            if not premise or not hypothesis:
                continue

            # Convert to a reasoning prompt
            prompt_text = (
                f"नीचे दो वाक्य दिए गए हैं। बताओ कि दूसरा वाक्य पहले वाक्य से "
                f"निकलता है (entailment), विरोधी है (contradiction), या संबंधित नहीं है (neutral)।\n\n"
                f"वाक्य 1: {premise}\n"
                f"वाक्य 2: {hypothesis}\n\n"
                f"अपना जवाब दो और explain करो क्यों।"
            )

            prompts.append({
                "id": f"HI-XNLI-{idx+1:03d}",
                "text": prompt_text,
                "category": "creative_reasoning",
                "language": "hi",
                "description": "XNLI Hindi - Natural language inference",
                "reference_answer": label_map.get(label, "neutral"),
                "source": "facebook/xnli",
                "metadata": {
                    "difficulty": "hard",
                    "topic": "reasoning",
                    "dataset": "XNLI",
                    "label": label_map.get(label, "neutral")
                }
            })

        print(f"  Downloaded {len(prompts)} XNLI Hindi prompts")
        return prompts

    except Exception as e:
        print(f"  Error downloading XNLI: {e}")
        return []


def create_hindi_instruction_prompts() -> List[Dict[str, Any]]:
    """
    Create additional Hindi instruction-following prompts.

    These are manually curated to test specific instruction-following capabilities
    in natural, conversational Hindi.
    """
    print("Creating Hindi instruction-following prompts...")

    prompts = [
        # Format constraints
        {
            "id": "HI-IF-101",
            "text": "भारत के पांच सबसे बड़े राज्यों के नाम बताओ। सिर्फ नाम लिखो, कोई explanation नहीं।",
            "category": "instruction_following",
            "language": "hi",
            "description": "List format with no explanation constraint",
            "expected_format": "5 state names only, no explanation",
            "metadata": {"difficulty": "easy", "topic": "geography"}
        },
        {
            "id": "HI-IF-102",
            "text": "एक छोटी कविता लिखो जिसमें 'आसमान', 'तारे', और 'सपने' तीनों शब्द हों। कविता 4 lines की होनी चाहिए।",
            "category": "instruction_following",
            "language": "hi",
            "description": "Poetry with specific words and line count",
            "expected_format": "4-line poem containing 'आसमान', 'तारे', 'सपने'",
            "metadata": {"difficulty": "medium", "topic": "creative_writing"}
        },
        {
            "id": "HI-IF-103",
            "text": "मुझे exactly 50 शब्दों में बताओ कि exercise करना क्यों जरूरी है। ना कम, ना ज्यादा।",
            "category": "instruction_following",
            "language": "hi",
            "description": "Exact word count constraint",
            "expected_format": "Exactly 50 words about exercise importance",
            "metadata": {"difficulty": "hard", "topic": "health"}
        },
        {
            "id": "HI-IF-104",
            "text": "नीचे दी गई list को alphabetically sort करो और numbered list में लिखो:\nसेब, आम, केला, अंगूर, संतरा",
            "category": "instruction_following",
            "language": "hi",
            "description": "Sorting and formatting task",
            "expected_format": "Numbered list in Hindi alphabetical order",
            "metadata": {"difficulty": "medium", "topic": "language"}
        },
        {
            "id": "HI-IF-105",
            "text": "एक recipe लिखो जिसमें: (1) Ingredients की list हो, (2) Step-by-step instructions हों (numbered), (3) Total time लगे। Recipe simple होनी चाहिए।",
            "category": "instruction_following",
            "language": "hi",
            "description": "Multi-section structured format",
            "expected_format": "Recipe with ingredients list, numbered steps, and time",
            "metadata": {"difficulty": "medium", "topic": "cooking"}
        },
        {
            "id": "HI-IF-106",
            "text": "मुझे तीन paragraphs में mobile phones के बारे में बताओ:\n1. पहला paragraph: फायदे (pros)\n2. दूसरा paragraph: नुकसान (cons)\n3. तीसरा paragraph: conclusion\n\nहर paragraph 30-40 शब्दों का हो।",
            "category": "instruction_following",
            "language": "hi",
            "description": "Structured paragraphs with word limits",
            "expected_format": "3 paragraphs (pros, cons, conclusion), 30-40 words each",
            "metadata": {"difficulty": "hard", "topic": "technology"}
        },
        {
            "id": "HI-IF-107",
            "text": "एक formal letter लिखो अपने school principal को जिसमें 2 दिन की छुट्टी मांगो। Letter में: date, subject, salutation, body, और signature होनी चाहिए।",
            "category": "instruction_following",
            "language": "hi",
            "description": "Formal letter format compliance",
            "expected_format": "Formal Hindi letter with all required components",
            "metadata": {"difficulty": "medium", "topic": "communication"}
        },
        {
            "id": "HI-IF-108",
            "text": "नीचे दिए गए sentence को passive voice में बदलो:\n'राम ने खाना खाया।'",
            "category": "instruction_following",
            "language": "hi",
            "description": "Grammar transformation task",
            "expected_format": "Passive voice: 'राम द्वारा खाना खाया गया।'",
            "reference_answer": "राम द्वारा खाना खाया गया।",
            "metadata": {"difficulty": "easy", "topic": "grammar"}
        },
        {
            "id": "HI-IF-109",
            "text": "एक dialog लिखो जिसमें दो दोस्त cricket match के बारे में बात कर रहे हों। Dialog में कम से कम 6 exchanges (3-3 दोनों तरफ से) होने चाहिए।",
            "category": "instruction_following",
            "language": "hi",
            "description": "Dialogue format with minimum exchanges",
            "expected_format": "Dialogue with at least 6 exchanges about cricket",
            "metadata": {"difficulty": "medium", "topic": "sports"}
        },
        {
            "id": "HI-IF-110",
            "text": "इस paragraph को summarize करो exactly 2 sentences में:\n\n'भारत एक विविधताओं से भरा देश है। यहां अलग-अलग धर्म, भाषाएं, और संस्कृतियां हैं। उत्तर में हिमालय है तो दक्षिण में समुद्र। हर राज्य की अपनी पहचान है। यही विविधता भारत को unique बनाती है।'",
            "category": "instruction_following",
            "language": "hi",
            "description": "Summarization with exact sentence count",
            "expected_format": "Exactly 2-sentence summary",
            "metadata": {"difficulty": "medium", "topic": "culture"}
        },
    ]

    print(f"  Created {len(prompts)} instruction-following prompts")
    return prompts


def create_hindi_factual_prompts() -> List[Dict[str, Any]]:
    """
    Create factual QA prompts for Hindi with verifiable answers.
    """
    print("Creating Hindi factual QA prompts...")

    prompts = [
        {
            "id": "HI-FQ-101",
            "text": "भारत की राजधानी क्या है और इसे कब राजधानी बनाया गया था?",
            "category": "factual_qa",
            "language": "hi",
            "description": "Basic geography fact",
            "reference_answer": "भारत की राजधानी नई दिल्ली है। इसे 1911 में कलकत्ता की जगह राजधानी बनाया गया था और 1931 में officially shift हुआ।",
            "metadata": {"difficulty": "easy", "topic": "geography"}
        },
        {
            "id": "HI-FQ-102",
            "text": "चंद्रयान-3 कब launch हुआ था और इसने क्या achieve किया?",
            "category": "factual_qa",
            "language": "hi",
            "description": "Recent Indian space achievement",
            "reference_answer": "चंद्रयान-3 14 जुलाई 2023 को launch हुआ। 23 अगस्त 2023 को इसका Vikram lander चंद्रमा के दक्षिणी ध्रुव पर safely land हुआ, जिससे भारत ऐसा करने वाला पहला देश बना।",
            "metadata": {"difficulty": "medium", "topic": "science"}
        },
        {
            "id": "HI-FQ-103",
            "text": "DNA का full form क्या है और यह किस काम आता है?",
            "category": "factual_qa",
            "language": "hi",
            "description": "Biology basic concept",
            "reference_answer": "DNA का full form Deoxyribonucleic Acid है। यह genetic information store करता है जो parents से बच्चों में transfer होती है। DNA में genes होते हैं जो हमारी body के traits decide करते हैं।",
            "metadata": {"difficulty": "easy", "topic": "biology"}
        },
        {
            "id": "HI-FQ-104",
            "text": "भारत में कितने राज्य और केंद्र शासित प्रदेश हैं? 2024 के अनुसार बताओ।",
            "category": "factual_qa",
            "language": "hi",
            "description": "Current political geography",
            "reference_answer": "भारत में 28 राज्य और 8 केंद्र शासित प्रदेश हैं (2024 के अनुसार)।",
            "metadata": {"difficulty": "easy", "topic": "civics"}
        },
        {
            "id": "HI-FQ-105",
            "text": "पानी का chemical formula क्या है और पानी किस-किस रूप में पाया जाता है?",
            "category": "factual_qa",
            "language": "hi",
            "description": "Chemistry basic concept",
            "reference_answer": "पानी का chemical formula H2O है (2 hydrogen atoms + 1 oxygen atom)। पानी तीन रूपों में पाया जाता है: ठोस (बर्फ), तरल (पानी), और गैस (भाप/water vapor)।",
            "metadata": {"difficulty": "easy", "topic": "chemistry"}
        },
        {
            "id": "HI-FQ-106",
            "text": "महाभारत में कितने दिन युद्ध हुआ था और इसमें कौन-कौन सी main armies थीं?",
            "category": "factual_qa",
            "language": "hi",
            "description": "Indian mythology/history",
            "reference_answer": "महाभारत का युद्ध 18 दिन चला। इसमें दो main armies थीं: पांडव सेना (अर्जुन, भीम, युधिष्ठिर के साथ) और कौरव सेना (दुर्योधन के नेतृत्व में)। श्री कृष्ण पांडवों के सारथी थे।",
            "metadata": {"difficulty": "medium", "topic": "mythology"}
        },
        {
            "id": "HI-FQ-107",
            "text": "भारत में सबसे लंबी नदी कौन सी है और यह कहां से निकलती है?",
            "category": "factual_qa",
            "language": "hi",
            "description": "Geography - Rivers",
            "reference_answer": "भारत की सबसे लंबी नदी गंगा है (2,525 km)। यह उत्तराखंड में गंगोत्री ग्लेशियर से निकलती है और बंगाल की खाड़ी में मिलती है।",
            "metadata": {"difficulty": "easy", "topic": "geography"}
        },
        {
            "id": "HI-FQ-108",
            "text": "Computer में RAM और ROM में क्या difference है?",
            "category": "factual_qa",
            "language": "hi",
            "description": "Computer basics",
            "reference_answer": "RAM (Random Access Memory) temporary memory है जो power off होने पर data खो देती है। ROM (Read Only Memory) permanent memory है जिसका data power off होने पर भी रहता है। RAM में programs run होते हैं, ROM में boot instructions store होती हैं।",
            "metadata": {"difficulty": "medium", "topic": "technology"}
        },
        {
            "id": "HI-FQ-109",
            "text": "भारत का national anthem किसने लिखा और इसे कब adopt किया गया?",
            "category": "factual_qa",
            "language": "hi",
            "description": "National symbols",
            "reference_answer": "भारत का राष्ट्रगान 'जन गण मन' रवींद्रनाथ टैगोर ने लिखा था। इसे 24 जनवरी 1950 को भारत के राष्ट्रगान के रूप में adopt किया गया।",
            "metadata": {"difficulty": "easy", "topic": "national_symbols"}
        },
        {
            "id": "HI-FQ-110",
            "text": "Solar system में कितने planets हैं? उनके नाम सूर्य से दूरी के order में बताओ।",
            "category": "factual_qa",
            "language": "hi",
            "description": "Astronomy basics",
            "reference_answer": "Solar system में 8 planets हैं। सूर्य से दूरी के order में: बुध (Mercury), शुक्र (Venus), पृथ्वी (Earth), मंगल (Mars), बृहस्पति (Jupiter), शनि (Saturn), अरुण (Uranus), वरुण (Neptune)।",
            "metadata": {"difficulty": "medium", "topic": "astronomy"}
        },
    ]

    print(f"  Created {len(prompts)} factual QA prompts")
    return prompts


def create_hindi_reasoning_prompts() -> List[Dict[str, Any]]:
    """
    Create creative reasoning prompts for Hindi.
    """
    print("Creating Hindi reasoning prompts...")

    prompts = [
        {
            "id": "HI-CR-101",
            "text": "अगर धरती पर gravity नहीं होती तो हमारी daily life कैसी होती? कम से कम 5 changes बताओ।",
            "category": "creative_reasoning",
            "language": "hi",
            "description": "Counterfactual reasoning - physics",
            "metadata": {"difficulty": "medium", "topic": "physics_hypothetical"}
        },
        {
            "id": "HI-CR-102",
            "text": "एक puzzle solve करो: एक कमरे में 3 switches हैं और दूसरे कमरे में 3 bulbs। तुम एक बार ही दूसरे कमरे में जा सकते हो। कैसे पता करोगे कि कौन सा switch किस bulb का है?",
            "category": "creative_reasoning",
            "language": "hi",
            "description": "Logic puzzle",
            "reference_answer": "पहले switch को 10 मिनट के लिए on करो, फिर off करो। दूसरा switch on करो और तीसरा off रखो। अब दूसरे कमरे में जाओ: जो bulb जल रहा है वो दूसरे switch का है, जो गर्म है पर बंद है वो पहले का है, और जो ठंडा और बंद है वो तीसरे का है।",
            "metadata": {"difficulty": "hard", "topic": "logic_puzzle"}
        },
        {
            "id": "HI-CR-103",
            "text": "तुम एक time traveler हो और 100 साल पीछे जा सकते हो। तुम किस event को बदलना चाहोगे और क्यों? इसके possible consequences भी बताओ।",
            "category": "creative_reasoning",
            "language": "hi",
            "description": "Hypothetical scenario with consequence analysis",
            "metadata": {"difficulty": "hard", "topic": "hypothetical"}
        },
        {
            "id": "HI-CR-104",
            "text": "एक छोटे से गांव में सिर्फ 2 barbers हैं। एक की दुकान बहुत साफ है और वो खुद बहुत well-groomed है। दूसरे की दुकान गंदी है और उसके बाल भी बिखरे हैं। तुम्हें haircut लेना है - किसके पास जाओगे और क्यों?",
            "category": "creative_reasoning",
            "language": "hi",
            "description": "Logic puzzle - barber paradox",
            "reference_answer": "गंदी दुकान वाले barber के पास जाना चाहिए। क्योंकि गांव में सिर्फ 2 barbers हैं, तो वो एक-दूसरे के बाल काटते होंगे। साफ दुकान वाले के बाल गंदी दुकान वाले ने काटे होंगे (जो अच्छे लग रहे हैं), मतलब गंदी दुकान वाला बेहतर barber है।",
            "metadata": {"difficulty": "hard", "topic": "logic_puzzle"}
        },
        {
            "id": "HI-CR-105",
            "text": "एक कहानी पूरी करो: 'जब मैंने वो पुराना संदूक खोला, तो अंदर से एक चिट्ठी निकली जो 50 साल पहले लिखी गई थी। चिट्ठी में लिखा था...'",
            "category": "creative_reasoning",
            "language": "hi",
            "description": "Story continuation",
            "metadata": {"difficulty": "medium", "topic": "creative_writing"}
        },
        {
            "id": "HI-CR-106",
            "text": "एक नई technology invent करो जो education को बेहतर बना सके। इसका नाम, काम करने का तरीका, और 3 फायदे बताओ।",
            "category": "creative_reasoning",
            "language": "hi",
            "description": "Innovation and problem solving",
            "metadata": {"difficulty": "medium", "topic": "innovation"}
        },
        {
            "id": "HI-CR-107",
            "text": "अगर जानवर बोल पाते तो क्या होता? इस scenario के positive और negative दोनों aspects discuss करो।",
            "category": "creative_reasoning",
            "language": "hi",
            "description": "Counterfactual with balanced analysis",
            "metadata": {"difficulty": "medium", "topic": "hypothetical"}
        },
        {
            "id": "HI-CR-108",
            "text": "एक math puzzle: एक train 100 km/hr की speed से चल रही है। एक मक्खी 150 km/hr से उड़ती है। अगर दो trains 200 km दूर से एक-दूसरे की तरफ आ रही हैं और मक्खी पहली train से दूसरी तक, फिर वापस, continuously उड़ती रहती है जब तक trains नहीं मिलतीं - मक्खी total कितना distance cover करेगी?",
            "category": "creative_reasoning",
            "language": "hi",
            "description": "Math word problem",
            "reference_answer": "दोनों trains की relative speed = 100 + 100 = 200 km/hr। वो 200 km cover करने में 1 hour लेंगी। इस 1 hour में मक्खी 150 km/hr से उड़ती रहेगी, तो total distance = 150 km।",
            "metadata": {"difficulty": "hard", "topic": "math_puzzle"}
        },
        {
            "id": "HI-CR-109",
            "text": "तुम्हारे पास 8 identical दिखने वाले balls हैं। एक ball थोड़ी भारी है। तुम्हारे पास एक balance scale है। minimum कितनी बार तौलना होगा भारी ball ढूंढने के लिए?",
            "category": "creative_reasoning",
            "language": "hi",
            "description": "Classic weighing puzzle",
            "reference_answer": "सिर्फ 2 बार तौलना होगा। पहली बार: 3-3 balls तौलो। अगर equal हैं तो भारी ball बची 2 में है (एक और weighing)। अगर एक side भारी है तो उस group की 3 में से 1-1 तौलो।",
            "metadata": {"difficulty": "hard", "topic": "logic_puzzle"}
        },
        {
            "id": "HI-CR-110",
            "text": "Climate change से लड़ने के लिए एक ऐसा solution suggest करो जो affordable हो और developing countries में implement हो सके। अपने idea के pros और cons भी बताओ।",
            "category": "creative_reasoning",
            "language": "hi",
            "description": "Problem solving with constraints",
            "metadata": {"difficulty": "hard", "topic": "environment"}
        },
    ]

    print(f"  Created {len(prompts)} reasoning prompts")
    return prompts


def download_ifeval_english(max_samples: int = 50) -> List[Dict[str, Any]]:
    """
    Download Google's IFEval dataset for English instruction following.
    """
    print("Downloading Google IFEval English...")

    try:
        dataset = load_dataset("google/IFEval", split="train")

        prompts = []

        for idx, item in enumerate(dataset):
            if len(prompts) >= max_samples:
                break

            prompt_text = item.get("prompt", "").strip()
            instruction_ids = item.get("instruction_id_list", [])
            kwargs = item.get("kwargs", [])

            if not prompt_text:
                continue

            # Determine difficulty based on number of constraints
            num_constraints = len(instruction_ids)
            if num_constraints <= 1:
                difficulty = "easy"
            elif num_constraints <= 2:
                difficulty = "medium"
            else:
                difficulty = "hard"

            prompts.append({
                "id": f"EN-IF-{idx+1:03d}",
                "text": prompt_text,
                "category": "instruction_following",
                "language": "en",
                "description": f"IFEval - {num_constraints} verifiable constraint(s)",
                "source": "google/IFEval",
                "instruction_ids": instruction_ids,
                "verification_kwargs": kwargs,
                "metadata": {
                    "difficulty": difficulty,
                    "topic": "instruction_following",
                    "dataset": "IFEval",
                    "num_constraints": num_constraints
                }
            })

        print(f"  Downloaded {len(prompts)} IFEval prompts")
        return prompts

    except Exception as e:
        print(f"  Error downloading IFEval: {e}")
        return []


def download_truthfulqa_english(max_samples: int = 40) -> List[Dict[str, Any]]:
    """
    Download TruthfulQA dataset for factuality evaluation.
    """
    print("Downloading TruthfulQA English...")

    try:
        dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")

        prompts = []

        for idx, item in enumerate(dataset):
            if len(prompts) >= max_samples:
                break

            question = item.get("question", "").strip()
            best_answer = item.get("best_answer", "").strip()
            correct_answers = item.get("correct_answers", [])
            incorrect_answers = item.get("incorrect_answers", [])
            category = item.get("category", "general")

            if not question:
                continue

            # Combine correct answers for reference
            reference = best_answer
            if correct_answers and not reference:
                reference = correct_answers[0]

            prompts.append({
                "id": f"EN-TQ-{idx+1:03d}",
                "text": question,
                "category": "factual_qa",
                "language": "en",
                "description": f"TruthfulQA - {category}",
                "reference_answer": reference,
                "source": "truthfulqa/truthful_qa",
                "incorrect_answers": incorrect_answers[:3] if incorrect_answers else [],
                "metadata": {
                    "difficulty": "hard",
                    "topic": category,
                    "dataset": "TruthfulQA"
                }
            })

        print(f"  Downloaded {len(prompts)} TruthfulQA prompts")
        return prompts

    except Exception as e:
        print(f"  Error downloading TruthfulQA: {e}")
        return []


def merge_with_existing(new_prompts: List[Dict], existing_file: Path) -> List[Dict]:
    """
    Merge new prompts with existing ones, avoiding duplicates.
    """
    if existing_file.exists():
        with open(existing_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            existing_prompts = existing_data.get("prompts", [])
    else:
        existing_prompts = []

    existing_ids = {p["id"] for p in existing_prompts}

    for prompt in new_prompts:
        if prompt["id"] not in existing_ids:
            existing_prompts.append(prompt)
            existing_ids.add(prompt["id"])

    return existing_prompts


def save_dataset(prompts: List[Dict], output_path: Path, name: str, language: str, description: str):
    """
    Save prompts to JSON file in project format.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = {
        "name": name,
        "version": "2.0.0",
        "description": description,
        "language": language,
        "total_prompts": len(prompts),
        "categories": {
            "instruction_following": len([p for p in prompts if p["category"] == "instruction_following"]),
            "factual_qa": len([p for p in prompts if p["category"] == "factual_qa"]),
            "creative_reasoning": len([p for p in prompts if p["category"] == "creative_reasoning"]),
        },
        "sources": list(set(p.get("source", "manual") for p in prompts)),
        "prompts": prompts
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(prompts)} prompts to {output_path}")
    print(f"  - Instruction Following: {dataset['categories']['instruction_following']}")
    print(f"  - Factual QA: {dataset['categories']['factual_qa']}")
    print(f"  - Creative Reasoning: {dataset['categories']['creative_reasoning']}")


def main():
    parser = argparse.ArgumentParser(description="Download LLM evaluation datasets")
    parser.add_argument(
        "--language", "-l",
        choices=["en", "hi", "all"],
        default="all",
        help="Language to download (en=English, hi=Hindi, all=both)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/prompts",
        help="Output directory for datasets"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Maximum samples per dataset"
    )

    args = parser.parse_args()
    output_dir = Path(args.output)

    if not HF_AVAILABLE:
        print("\nError: 'datasets' library required. Install with: pip install datasets")
        print("\nFalling back to manually curated prompts only...")

    # Download Hindi datasets
    if args.language in ["hi", "all"]:
        print("\n" + "="*60)
        print("DOWNLOADING HINDI DATASETS")
        print("="*60)

        hindi_prompts = []

        # Download from HuggingFace if available
        if HF_AVAILABLE:
            hindi_prompts.extend(download_indicqa_hindi(args.max_samples))
            hindi_prompts.extend(download_xnli_hindi(20))

        # Add manually curated prompts
        hindi_prompts.extend(create_hindi_instruction_prompts())
        hindi_prompts.extend(create_hindi_factual_prompts())
        hindi_prompts.extend(create_hindi_reasoning_prompts())

        # Save expanded Hindi dataset
        save_dataset(
            hindi_prompts,
            output_dir / "hindi_benchmark_expanded.json",
            name="Hindi Benchmark Prompts (Expanded)",
            language="hi",
            description="Comprehensive Hindi LLM evaluation prompts from IndicQA, XNLI, and manual curation. Natural, conversational Hindi."
        )

    # Download English datasets
    if args.language in ["en", "all"]:
        print("\n" + "="*60)
        print("DOWNLOADING ENGLISH DATASETS")
        print("="*60)

        english_prompts = []

        if HF_AVAILABLE:
            english_prompts.extend(download_ifeval_english(args.max_samples))
            english_prompts.extend(download_truthfulqa_english(40))

        if english_prompts:
            save_dataset(
                english_prompts,
                output_dir / "english_benchmark_expanded.json",
                name="English Benchmark Prompts (Expanded)",
                language="en",
                description="Comprehensive English LLM evaluation prompts from IFEval and TruthfulQA."
            )

    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)
    print(f"\nDatasets saved to: {output_dir}")
    print("\nTo use the expanded datasets:")
    print(f"  python main.py analyze --prompts {output_dir}/hindi_benchmark_expanded.json")


if __name__ == "__main__":
    main()
