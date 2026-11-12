"""Convert the FICTIONAL rehearsal persona into the data contract of brasa.data.

    uv run python scripts/make_rehearsal.py                     # writes data/rehearsal/{kb,qa}
    uv run python scripts/make_rehearsal.py --k 4 --out runs/rehearsal-k4     # fewer training paraphrases

"Téo Arantes" does not exist. The persona (rehearsal_persona.py, rehearsal_persona_extra.py) was written
to rehearse the whole pipeline before the real corpus exists, and to measure how many question
paraphrases a fact needs before held-out phrasings are answered.

Per fact and language there are 14 hand-written questions. A fixed shuffle of the first 8 keeps 2 as
held-out TEST paraphrases and 1 for VALIDATION; the other 5 plus 6 extra ones form the training pool, of
which the first --k are used (default: all 11). Every training question also gets one copy wrapped in a
greeting or a polite ending. Out-of-scope questions: frames x fillers for training, unseen fillers and
unseen frames for test. Multi-fact and small-talk records are defined below.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import rehearsal_persona as P
from rehearsal_persona_extra import EXTRA

# slug -> (type, title_pt, title_en, aliases, fact ids)
TOPICS = {
    "profile": ("person", "Perfil", "Profile", ["Téo Arantes"],
                ["who", "location", "spoken", "hobbies", "first_program"]),
    "career": ("experience", "Carreira", "Career", ["Lumen Forge", "Brisa Logistics"],
               ["experience", "education", "current_work", "previous_job", "certification", "talks"]),
    "skills": ("skill", "Habilidades", "Skills", ["stack"],
               ["languages", "frameworks", "cloud", "databases", "tools", "mlops", "llm", "favourite_area"]),
    "sentinela": ("project", "Sentinela", "Sentinela", ["fintech", "fraude", "fraud"], ["sentinela"]),
    "notaclara": ("project", "NotaClara", "NotaClara", ["OCR", "notas fiscais", "invoices"], ["notaclara"]),
    "tessela": ("project", "Tessela", "Tessela", ["código aberto", "open source"], ["open_source"]),
    "services": ("service", "Serviços", "Services", ["consultoria", "consulting"],
                 ["services", "availability", "remote", "pricing"]),
    "contact": ("contact", "Contato", "Contact", ["e-mail", "email"], ["contact", "timezone"]),
    "assistant": ("meta", "Este assistente", "This assistant", ["chatbot", "modelo", "model"],
                  ["assistant", "model_size"]),
}

# The numbers and names an answer must contain: fact id -> spans for both languages, or (en, pt).
KEY_SPANS = {
    "who": ["Téo Arantes"],
    "location": ["Curitiba"],
    "experience": (["nine"], ["nove"]),
    "languages": ["Python", "TypeScript", "Rust"],
    "frameworks": ["PyTorch", "scikit-learn"],
    "education": (["Computer Engineering", "Federal University of Paraná"],
                  ["Engenharia de Computação", "Universidade Federal do Paraná"]),
    "current_work": ["Lumen Forge"],
    "previous_job": (["four", "Brisa Logistics"], ["quatro", "Brisa Logistics"]),
    "sentinela": ["Sentinela", "40"],
    "notaclara": ["NotaClara"],
    "assistant": ["PyTorch"],
    "model_size": (["four million"], ["quatro milhões"]),
    "open_source": ["Tessela"],
    "spoken": (["Portuguese", "English", "Spanish"], ["português", "inglês", "espanhol"]),
    "contact": ["hello@lumenforge.example"],
    "availability": (["March"], ["março"]),
    "services": (["consulting"], ["consultoria"]),
    "remote": (["Brazil", "Europe", "United States"], ["Brasil", "Europa", "Estados Unidos"]),
    "pricing": (["fixed-price"], ["preço fixo"]),
    "cloud": ["AWS", "Google Cloud", "Docker", "Kubernetes"],
    "databases": ["PostgreSQL", "Redis", "ClickHouse"],
    "certification": ["AWS Machine Learning Specialty"],
    "hobbies": (["cello"], ["violoncelo"]),
    "talks": ["Python Brasil 2023"],
    "timezone": (["UTC minus three"], ["UTC menos três"]),
    "first_program": (["twelve", "BASIC"], ["doze", "BASIC"]),
    "tools": ["Neovim", "Git"],
    "favourite_area": (["efficient inference"], ["inferência eficiente"]),
    "mlops": ["MLflow", "GitHub Actions"],
    "llm": (["retrieval"], ["busca"]),
}

# Questions that need two facts. The answer is the two fact sentences, in this order.
# name -> (fact ids, [4 train, 1 val, 1 test] in English, the same in Portuguese)
MULTI = {
    "infra": (["cloud", "databases"],
              ["What is his full infrastructure and database stack?",
               "Which cloud and database technologies does Téo use?",
               "Tell me about his cloud platforms and his databases.",
               "What does he use for deployment and for storing data?",
               "Describe Téo's cloud setup and data stores.",
               "Which clouds and which databases does he work with?"],
              ["Qual é a stack completa de infraestrutura e bancos de dados dele?",
               "Quais tecnologias de nuvem e de banco de dados o Téo usa?",
               "Me fale sobre as plataformas de nuvem e os bancos de dados dele.",
               "O que ele usa para deploy e para armazenar dados?",
               "Descreva a nuvem e os armazenamentos de dados do Téo.",
               "Com quais nuvens e quais bancos de dados ele trabalha?"]),
    "code": (["languages", "frameworks"],
             ["Which programming languages and ML frameworks does Téo use?",
              "What languages and machine learning libraries does he work with?",
              "Tell me his programming languages and his deep learning tools.",
              "What does he code in and what does he train models with?",
              "Give me his languages and frameworks.",
              "Which languages and which ML frameworks is he good at?"],
             ["Quais linguagens de programação e frameworks de ML o Téo usa?",
              "Com quais linguagens e bibliotecas de machine learning ele trabalha?",
              "Me diga as linguagens de programação e as ferramentas de deep learning dele.",
              "Em que ele programa e com o que ele treina modelos?",
              "Me passe as linguagens e os frameworks dele.",
              "Em quais linguagens e em quais frameworks de ML ele é bom?"]),
    "jobs": (["current_work", "previous_job"],
             ["What is his current job and where did he work before?",
              "Tell me about Téo's current and previous work.",
              "Where does he work now and where was he before that?",
              "Summarize his career: present and past jobs.",
              "What does he do today and what did he do earlier?",
              "Describe his present company and his former employer."],
             ["Qual é o trabalho atual dele e onde ele trabalhou antes?",
              "Me fale sobre o trabalho atual e o anterior do Téo.",
              "Onde ele trabalha hoje e onde estava antes disso?",
              "Resuma a carreira dele: empregos atual e passado.",
              "O que ele faz hoje e o que fazia antes?",
              "Descreva a empresa atual e o antigo empregador dele."]),
    "projects": (["sentinela", "notaclara"],
                 ["What are Téo's main projects?",
                  "Tell me about Sentinela and NotaClara.",
                  "Which projects has he built?",
                  "Describe his fraud detection and invoice reading projects.",
                  "Give me an overview of his two main projects.",
                  "What did he build for fraud and for invoices?"],
                 ["Quais são os principais projetos do Téo?",
                  "Me fale sobre o Sentinela e o NotaClara.",
                  "Quais projetos ele já construiu?",
                  "Descreva os projetos de detecção de fraude e de leitura de notas fiscais dele.",
                  "Me dê uma visão geral dos dois principais projetos dele.",
                  "O que ele construiu para fraude e para notas fiscais?"]),
    "reach": (["contact", "timezone"],
              ["How can I contact Téo and how fast does he reply?",
               "What is his email and his time zone?",
               "How do I reach him and when will he answer?",
               "Give me his contact and his usual response time.",
               "Where do I write to him and how long until he responds?",
               "What is the best way to reach him and how quickly does he answer?"],
              ["Como posso entrar em contato com o Téo e em quanto tempo ele responde?",
               "Qual é o e-mail e o fuso horário dele?",
               "Como eu falo com ele e quando ele vai responder?",
               "Me passe o contato e o tempo de resposta dele.",
               "Para onde eu escrevo e quanto tempo até ele responder?",
               "Qual é a melhor forma de falar com ele e com que rapidez ele responde?"]),
}

# Small talk: intent -> (answer en, answer pt, [4 train, 1 val, 1 test] en, the same pt)
CHITCHAT = {
    "greeting": (
        "Hello! I am a small language model that answers questions about Téo Arantes. What would you like to know?",
        "Olá! Sou um pequeno modelo de linguagem que responde perguntas sobre o Téo Arantes. O que você gostaria "
        "de saber?",
        ["Hi", "Hello", "Hey there", "Good morning", "Hey!", "Hello there"],
        ["Oi", "Olá", "E aí", "Bom dia", "Oi, tudo bem?", "Olá, tudo bem?"]),
    "thanks": (
        "You're welcome! Ask me anything else about Téo.",
        "De nada! Pergunte o que mais quiser sobre o Téo.",
        ["Thanks", "Thank you", "Thanks a lot", "Thank you very much", "Great, thanks!", "Thanks for the help"],
        ["Obrigado", "Valeu", "Muito obrigado", "Obrigada", "Legal, obrigado!", "Obrigado pela ajuda"]),
    "bye": (
        "Goodbye! Thanks for visiting.",
        "Tchau! Obrigado pela visita.",
        ["Bye", "Goodbye", "See you", "See you later", "Bye bye", "That's all, bye"],
        ["Tchau", "Até logo", "Até mais", "Adeus", "Tchau tchau", "É só isso, tchau"]),
    "help": (
        "You can ask me about Téo's projects, skills, experience, services and how to contact him.",
        "Você pode me perguntar sobre os projetos, as habilidades, a experiência, os serviços e o contato do Téo.",
        ["What can I ask you?", "What can you do?", "Help", "How can you help me?", "What do you know?",
         "What kind of questions can you answer?"],
        ["O que eu posso te perguntar?", "O que você sabe fazer?", "Ajuda", "Como você pode me ajudar?",
         "O que você sabe?", "Que tipo de pergunta você responde?"]),
}

SPLIT_OF_SLOT = ["train", "train", "train", "train", "val", "test"]


def spans(fid: str, lang: str) -> list[str]:
    s = KEY_SPANS[fid]
    return list(s if isinstance(s, list) else s[0 if lang == "en" else 1])


def wrap(q: str, lang: str, rng: random.Random) -> str:
    """A greeting in front of the question or a polite ending after it."""
    if rng.random() < 0.6:
        return rng.choice(P.PREFIXES[lang]) + q
    return q + rng.choice(P.SUFFIXES[lang])


def build(k: int, seed: int) -> tuple[list[dict], list[dict], dict[str, list[dict]]]:
    topic_of = {fid: slug for slug, (_, _, _, _, fids) in TOPICS.items() for fid in fids}
    answers = {fid: {"en": a_en, "pt": a_pt} for fid, a_en, a_pt, _, _ in P.FACTS}
    assert set(topic_of) == set(answers) == set(KEY_SPANS), "every fact needs a topic and key spans"

    topics = [{"slug": slug, "type": typ, "title_pt": pt, "title_en": en, "aliases": aliases}
              for slug, (typ, pt, en, aliases, _) in TOPICS.items()]
    facts = [{"id": fid, "topic": topic_of[fid], "subtopic": fid, "kind": "statement", "pt": a_pt, "en": a_en,
              "importance": 2 if fid in ("who", "contact", "current_work", "sentinela", "notaclara") else 1,
              "caveat": None}
             for fid, a_en, a_pt, _, _ in P.FACTS]

    split_rng, wrap_rng = random.Random(seed), random.Random(seed + 1)
    qa: dict[str, list[dict]] = {"facts": [], "multi": [], "chitchat": [], "oos": []}

    def add(file: str, rid: str, lang: str, q: str, a: str, fact_ids: list[str], kind: str, split: str,
            key_spans: list[str]) -> None:
        qa[file].append({"id": rid, "lang": lang, "q": q, "a": a, "facts": fact_ids, "kind": kind, "split": split,
                         "key_spans": key_spans})
        if split == "train":  # one wrapped copy of every training question
            qa[file].append({"id": rid + "-w", "lang": lang, "q": wrap(q, lang, wrap_rng), "a": a, "facts": fact_ids,
                             "kind": kind, "split": split, "key_spans": key_spans})

    for fid, _, _, q_en, q_pt in P.FACTS:
        kind = "meta" if topic_of[fid] == "assistant" else "single"
        for lang, questions in (("en", q_en), ("pt", q_pt)):
            order = questions[:]
            split_rng.shuffle(order)
            pool = (order[:5] + EXTRA[fid][0 if lang == "en" else 1])[:k]
            rows = [(q, "train") for q in pool] + [(order[5], "val")] + [(q, "test") for q in order[6:]]
            for n, (q, split) in enumerate(rows):
                add("facts", f"{fid}-{lang}-{n:02d}", lang, q, answers[fid][lang], [fid], kind, split,
                    spans(fid, lang))

    for name, (fids, q_en, q_pt) in MULTI.items():
        for lang, questions in (("en", q_en), ("pt", q_pt)):
            answer = " ".join(answers[f][lang] for f in fids)
            for n, q in enumerate(questions):
                add("multi", f"multi-{name}-{lang}-{n}", lang, q, answer, fids, "multi", SPLIT_OF_SLOT[n],
                    [s for f in fids for s in spans(f, lang)])

    for intent, (a_en, a_pt, q_en, q_pt) in CHITCHAT.items():
        for lang, answer, questions in (("en", a_en, q_en), ("pt", a_pt, q_pt)):
            for n, q in enumerate(questions):
                add("chitchat", f"chat-{intent}-{lang}-{n}", lang, q, answer, [], "chitchat", SPLIT_OF_SLOT[n], [])

    for lang in ("en", "pt"):
        n = 0
        for frame, fillers, held in P.OOS_TRAIN_FRAMES[lang]:
            for split, items in (("train", fillers), ("test", held)):
                for x in items:
                    add("oos", f"oos-{lang}-{n:03d}", lang, frame.format(x), P.REFUSAL[lang], [], "oos", split, [])
                    n += 1
        for i, q in enumerate(P.OOS_NEW_FRAMES[lang]):  # frames never seen in training
            add("oos", f"oos-{lang}-{n:03d}", lang, q, P.REFUSAL[lang], [], "oos", "test" if i < 12 else "val", [])
            n += 1
        known, held = P.UNKNOWN_PERSONAL[lang]  # personal details that are not in the knowledge base
        for i, (q, split) in enumerate([(q, "train") for q in known] + [(q, "test") for q in held[:6]]
                                       + [(q, "val") for q in held[6:]]):
            add("oos", f"unk-{lang}-{i:02d}", lang, q, P.UNKNOWN[lang], [], "oos", split, [])
    return topics, facts, qa


def write(out: str | Path, k: int = 11, seed: int = 1234) -> dict:
    """Write the corpus in the contract format and return the number of records per file."""
    topics, facts, qa = build(k, seed)
    out = Path(out)
    (out / "kb").mkdir(parents=True, exist_ok=True)
    (out / "qa").mkdir(parents=True, exist_ok=True)
    (out / "kb/topics.json").write_text(json.dumps(topics, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    def jsonl(rows: list[dict]) -> str:
        return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)

    (out / "kb/facts.jsonl").write_text(jsonl(facts), encoding="utf-8")
    for name, rows in qa.items():
        (out / "qa" / f"{name}.jsonl").write_text(jsonl(rows), encoding="utf-8")
    corpus = {"name": "rehearsal", "fictional": True, "training_paraphrases_per_fact_and_language": k, "seed": seed,
              "description": "Invented persona (Téo Arantes does not exist) used to rehearse the pipeline. "
                             "Nothing measured on it is a result of the released model."}
    (out / "corpus.json").write_text(json.dumps(corpus, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return {"facts": len(facts), "topics": len(topics), **{f"qa/{name}": len(rows) for name, rows in qa.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data" / "rehearsal"))
    ap.add_argument("--k", type=int, default=11, help="training paraphrases per fact and language (1..11)")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    print(f"wrote {args.out}: {write(args.out, args.k, args.seed)} (k={args.k})")


if __name__ == "__main__":
    main()
