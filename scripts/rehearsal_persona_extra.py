"""Six more hand-written question paraphrases per fact and language for the FICTIONAL persona (see persona.py).
They are only ever used for TRAINING (the held-out test paraphrases always come from the original eight), so the
paraphrase pool per fact can grow from 6 to 12 while the test set stays fixed.
"""

EXTRA = {
    "who": (
        ["Could you tell me who Téo is?", "I'd like to know more about Téo Arantes.", "Téo Arantes: who is he?",
         "Describe Téo in one sentence.", "What kind of developer is Téo?", "Whose website is this?"],
        ["Pode me dizer quem é o Téo?", "Queria saber mais sobre o Téo Arantes.", "Téo Arantes: quem é ele?",
         "Descreva o Téo em uma frase.", "Que tipo de desenvolvedor é o Téo?", "De quem é este site?"]),
    "location": (
        ["What city does Téo live in?", "Where is he from?", "Téo's city?", "Tell me where Téo is located.",
         "Does he live in Curitiba?", "Where in Brazil is he?"],
        ["Em que cidade o Téo mora?", "De onde ele é?", "Cidade do Téo?", "Me diga onde o Téo fica.",
         "Ele mora em Curitiba?", "Em que lugar do Brasil ele está?"]),
    "experience": (
        ["How long has he been in the software industry?", "How much experience does Téo have?",
         "For how many years has he built software?", "Téo's experience in years?", "Is he an experienced engineer?",
         "How long is his career so far?"],
        ["Há quanto tempo ele está na área de software?", "Quanta experiência o Téo tem?",
         "Por quantos anos ele já construiu software?", "Experiência do Téo em anos?", "Ele é um engenheiro experiente?",
         "Quanto tempo de carreira ele tem?"]),
    "languages": (
        ["What languages does he write code in?", "Is Python his main language?", "Does Téo use Rust or TypeScript?",
         "Téo's programming languages?", "Which language does he use the most?", "What can he code in?"],
        ["Em quais linguagens ele escreve código?", "Python é a linguagem principal dele?", "O Téo usa Rust ou TypeScript?",
         "Linguagens de programação do Téo?", "Qual linguagem ele mais usa?", "Em que ele sabe programar?"]),
    "frameworks": (
        ["What framework does he train models with?", "Does Téo use scikit-learn?", "Which deep learning library does Téo use?",
         "Téo's machine learning tools?", "Is he a PyTorch person?", "How does he train his models?"],
        ["Com qual framework ele treina modelos?", "O Téo usa scikit-learn?", "Qual biblioteca de deep learning o Téo usa?",
         "Ferramentas de machine learning do Téo?", "Ele é do time PyTorch?", "Como ele treina os modelos dele?"]),
    "education": (
        ["What is his degree in?", "Where did he graduate?", "Did he study Computer Engineering?", "Téo's university?",
         "Tell me about his studies.", "Where did you go to university?"],
        ["Em que ele é formado?", "Onde ele se formou?", "Ele estudou Engenharia de Computação?", "Universidade do Téo?",
         "Me fale sobre os estudos dele.", "Onde você fez faculdade?"]),
    "current_work": (
        ["What does he do for a living today?", "Does Téo run a studio?", "What is the name of his company?",
         "Where is he working currently?", "Téo's current role?", "Is he self-employed?"],
        ["O que ele faz da vida hoje?", "O Téo tem um estúdio?", "Qual é o nome da empresa dele?",
         "Onde ele está trabalhando no momento?", "Cargo atual do Téo?", "Ele trabalha por conta própria?"]),
    "previous_job": (
        ["Where was Téo before Lumen Forge?", "What did he do at Brisa Logistics?", "Téo's former employer?",
         "How long did he stay at his last job?", "Has he worked as a senior engineer?", "What jobs has he had?"],
        ["Onde o Téo estava antes da Lumen Forge?", "O que ele fazia na Brisa Logistics?", "Antigo empregador do Téo?",
         "Quanto tempo ele ficou no último emprego?", "Ele já trabalhou como engenheiro sênior?", "Que empregos ele já teve?"]),
    "sentinela": (
        ["What does Sentinela do?", "Explain the Sentinela system.", "Has he built real-time scoring systems?",
         "Which project scores card transactions?", "Sentinela project?", "Tell me about his work against fraud."],
        ["O que o Sentinela faz?", "Explique o sistema Sentinela.", "Ele já construiu sistemas de avaliação em tempo real?",
         "Qual projeto avalia transações de cartão?", "Projeto Sentinela?", "Me fale sobre o trabalho dele contra fraude."]),
    "notaclara": (
        ["Explain the NotaClara pipeline.", "Which project reads invoices?", "Has Téo fine-tuned a vision model?",
         "NotaClara project?", "What did he build for invoice processing?", "Does he work with document extraction?"],
        ["Explique o pipeline NotaClara.", "Qual projeto lê notas fiscais?", "O Téo já ajustou um modelo de visão?",
         "Projeto NotaClara?", "O que ele construiu para processar notas fiscais?", "Ele trabalha com extração de documentos?"]),
    "assistant": (
        ["What is this assistant?", "Is this GPT?", "Did Téo train you himself?", "What model is this?",
         "Are you a transformer?", "Who trained this chatbot?"],
        ["O que é este assistente?", "Isto é o GPT?", "O próprio Téo treinou você?", "Que modelo é este?",
         "Você é um transformer?", "Quem treinou este chatbot?"]),
    "model_size": (
        ["How many parameters do you have?", "Is the model running on my device?", "Model size?",
         "Does this chat need a server?", "Is my data leaving the browser?", "How small is this network?"],
        ["Quantos parâmetros você tem?", "O modelo está rodando no meu dispositivo?", "Tamanho do modelo?",
         "Este chat precisa de servidor?", "Meus dados saem do navegador?", "Quão pequena é esta rede?"]),
    "open_source": (
        ["What library does he maintain?", "Explain Tessela.", "Téo's open source work?",
         "Has he released a forecasting library?", "Where can I see his public code?", "Does he do open source?"],
        ["Qual biblioteca ele mantém?", "Explique a Tessela.", "Trabalho de código aberto do Téo?",
         "Ele lançou uma biblioteca de previsão?", "Onde posso ver o código público dele?", "Ele faz código aberto?"]),
    "spoken": (
        ["Does Téo speak Spanish?", "Which human languages does he know?", "Languages spoken by Téo?",
         "Can we have meetings in English?", "Does he speak Portuguese?", "Is he trilingual?"],
        ["O Téo fala espanhol?", "Quais línguas humanas ele conhece?", "Idiomas falados pelo Téo?",
         "Podemos fazer reuniões em inglês?", "Ele fala português?", "Ele é trilíngue?"]),
    "contact": (
        ["What is Téo's email address?", "How do I send him an email?", "Contact details?",
         "Where can I write to him?", "Give me his contact.", "How do I reach Téo about a project?"],
        ["Qual é o endereço de e-mail do Téo?", "Como eu mando um e-mail para ele?", "Dados de contato?",
         "Onde posso escrever para ele?", "Me passe o contato dele.", "Como falo com o Téo sobre um projeto?"]),
    "availability": (
        ["When is Téo free to start a project?", "Is he booked at the moment?", "Téo's availability?",
         "Could he start in March?", "Is he accepting consulting work?", "When could we begin working together?"],
        ["Quando o Téo está livre para começar um projeto?", "Ele está com a agenda cheia no momento?", "Disponibilidade do Téo?",
         "Ele poderia começar em março?", "Ele está aceitando trabalhos de consultoria?", "Quando poderíamos começar a trabalhar juntos?"]),
    "services": (
        ["What does Téo offer to clients?", "Does he do machine learning consulting?", "Téo's services?",
         "Can he train a custom model for me?", "Does he design backend architectures?", "What could I hire him for?"],
        ["O que o Téo oferece aos clientes?", "Ele faz consultoria em machine learning?", "Serviços do Téo?",
         "Ele pode treinar um modelo sob medida para mim?", "Ele desenha arquiteturas de backend?", "Para que eu poderia contratar ele?"]),
    "remote": (
        ["Is Téo a remote worker?", "Does he work with clients in the United States?", "Remote or on site?",
         "Where are the companies he works with?", "Can we work together from another country?", "Does he travel to clients?"],
        ["O Téo trabalha de forma remota?", "Ele trabalha com clientes nos Estados Unidos?", "Remoto ou presencial?",
         "Onde ficam as empresas com que ele trabalha?", "Podemos trabalhar juntos de outro país?", "Ele viaja até os clientes?"]),
    "pricing": (
        ["What does he charge for a project?", "Does he work with fixed prices?", "Téo's rates?",
         "How is a project priced?", "Is there a discovery phase?", "How much would it cost to hire him?"],
        ["Quanto ele cobra por um projeto?", "Ele trabalha com preço fixo?", "Valores do Téo?",
         "Como um projeto é precificado?", "Existe uma fase de descoberta?", "Quanto custaria contratar ele?"]),
    "cloud": (
        ["Does Téo use Google Cloud?", "What cloud does he work with?", "Téo's cloud stack?",
         "Does he deploy with Kubernetes?", "Which platforms does he deploy on?", "Is he experienced with AWS and Docker?"],
        ["O Téo usa Google Cloud?", "Com qual nuvem ele trabalha?", "Stack de nuvem do Téo?",
         "Ele faz deploy com Kubernetes?", "Em quais plataformas ele faz deploy?", "Ele tem experiência com AWS e Docker?"]),
    "databases": (
        ["Does Téo use Redis?", "What database does he like?", "Téo's databases?", "Which database does he use for analytics?",
         "Is he good with PostgreSQL?", "Where does he keep the data of his systems?"],
        ["O Téo usa Redis?", "De qual banco de dados ele gosta?", "Bancos de dados do Téo?", "Qual banco ele usa para análises?",
         "Ele é bom em PostgreSQL?", "Onde ele guarda os dados dos sistemas dele?"]),
    "certification": (
        ["Which certification does Téo hold?", "Does he have an AWS certificate?", "Téo's certifications?",
         "Is he certified in machine learning?", "Any official credentials?", "Has he been certified by a cloud provider?"],
        ["Qual certificação o Téo tem?", "Ele tem um certificado da AWS?", "Certificações do Téo?",
         "Ele é certificado em machine learning?", "Alguma credencial oficial?", "Ele foi certificado por algum provedor de nuvem?"]),
    "hobbies": (
        ["What does Téo like outside of programming?", "Does he climb?", "Téo's hobbies?", "Does he play the cello?",
         "What does he do to relax?", "Is he into coffee?"],
        ["Do que o Téo gosta fora da programação?", "Ele escala?", "Hobbies do Téo?", "Ele toca violoncelo?",
         "O que ele faz para relaxar?", "Ele curte café?"]),
    "talks": (
        ["Has he presented at any conference?", "What did he talk about at Python Brasil?", "Téo's talks?",
         "Did he speak at an event in 2023?", "Does he give lectures?", "Has Téo been a conference speaker?"],
        ["Ele já apresentou em alguma conferência?", "Sobre o que ele falou na Python Brasil?", "Palestras do Téo?",
         "Ele palestrou em algum evento em 2023?", "Ele dá palestras?", "O Téo já foi palestrante em conferência?"]),
    "timezone": (
        ["Which time zone does he work in?", "How soon does Téo answer?", "Téo's time zone?",
         "Will he reply within a day?", "What is his response time?", "Is his time zone compatible with Europe?"],
        ["Em qual fuso ele trabalha?", "Em quanto tempo o Téo responde?", "Fuso horário do Téo?",
         "Ele responde em até um dia?", "Qual é o tempo de resposta dele?", "O fuso dele é compatível com a Europa?"]),
    "first_program": (
        ["How old was Téo when he wrote his first program?", "What was the first thing he coded?", "Téo's first program?",
         "Did he start with BASIC?", "When did he write his first code?", "How did you start programming?"],
        ["Quantos anos o Téo tinha quando escreveu o primeiro programa?", "Qual foi a primeira coisa que ele programou?",
         "Primeiro programa do Téo?", "Ele começou com BASIC?", "Quando ele escreveu o primeiro código?",
         "Como você começou a programar?"]),
    "tools": (
        ["What text editor does he write code in?", "Is he a Neovim user?", "Téo's editor?",
         "What does he use for version control?", "Which tools does he use every day?", "What is his coding environment?"],
        ["Em qual editor de texto ele escreve código?", "Ele usa Neovim?", "Editor do Téo?",
         "O que ele usa para controle de versão?", "Quais ferramentas ele usa todo dia?", "Qual é o ambiente de programação dele?"]),
    "favourite_area": (
        ["What does Téo like most in machine learning?", "Is he interested in efficient inference?", "Téo's favourite topic?",
         "What field does he focus on?", "Does he like making small models fast?", "What is he best at?"],
        ["Do que o Téo mais gosta em machine learning?", "Ele se interessa por inferência eficiente?", "Assunto favorito do Téo?",
         "Em que área ele se concentra?", "Ele gosta de deixar modelos pequenos rápidos?", "No que ele é melhor?"]),
    "mlops": (
        ["What does he use to track experiments?", "Does Téo automate his training runs?", "Téo's MLOps stack?",
         "Which tool does he use for experiment tracking?", "How does he run CI for models?", "Is he familiar with MLflow?"],
        ["O que ele usa para acompanhar experimentos?", "O Téo automatiza os treinamentos?", "Stack de MLOps do Téo?",
         "Qual ferramenta ele usa para rastrear experimentos?", "Como ele roda CI para modelos?", "Ele conhece MLflow?"]),
    "llm": (
        ["Has Téo worked with LLMs?", "Does he fine-tune language models?", "Téo's experience with language models?",
         "Has he built search over private documents?", "Did he make models for support teams?", "Is he experienced in generative AI?"],
        ["O Téo já trabalhou com LLMs?", "Ele ajusta modelos de linguagem?", "Experiência do Téo com modelos de linguagem?",
         "Ele já construiu busca sobre documentos privados?", "Ele fez modelos para times de suporte?", "Ele tem experiência em IA generativa?"]),
}
