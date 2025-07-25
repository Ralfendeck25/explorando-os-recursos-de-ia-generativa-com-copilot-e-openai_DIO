# Veja **como criar um chatbot especializado no caso Epstein** usando técnicas de NLP e IA Generativa, com código passo a passo. 
### Esse é o mais complexo (e útil!) dos três tópicos.  


<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/f27f5a6d-a155-4413-b013-85bc494a79cb" />


## **Chatbot sobre os Epstein Files com LangChain + OpenAI**  
**Objetivo**: Um bot que responda perguntas baseadas nos documentos reais do caso, citando fontes.  

### **Tecnologias**:  
- **Framework**: `LangChain` (para gerenciamento de documentos e memória).  
- **Modelo de IA**: `GPT-4` ou `gpt-3.5-turbo` (via OpenAI API).  
- **Processamento de Texto**: `Spacy` ou `NLTK` para pré-processamento.  
- **Armazenamento**: `FAISS` (para busca semântica eficiente).  

## **Passo a Passo**  

### **1. Coleta e Pré-processamento dos Dados**  
Baixe documentos do caso Epstein em PDF ( [u.s._v._jeffrey_epstein_indictment.pdf](https://github.com/user-attachments/files/21423063/u.s._v._jeffrey_epstein_indictment.pdf). Converta para texto:  
```python
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

texto_epstein = extract_text_from_pdf("u.s._v._jeffrey_epstein_indictment.pdf")
```

### **2. Dividir o Texto em Pedaços (Chunks)**  
LangChain precisa de textos curtos para buscar informações relevantes.  
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Tamanho dos pedaços
    chunk_overlap=200  # Sobreposição para contexto
)
chunks = splitter.split_text(texto_epstein)
```

### **3. Criar um Banco de Dados de Busca Semântica**  
Use **embeddings** para transformar texto em vetores e permitir buscas inteligentes.  
```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

embeddings = OpenAIEmbeddings(openai_api_key="SUA_API_KEY")
db = FAISS.from_texts(chunks, embeddings)
```

### **4. Configurar o Chatbot com Memória**  
Para o bot lembrar o contexto da conversa:  
```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)  # Criativo, mas controlado

chatbot = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db.as_retriever(),
    memory=memory
)
```

### **5. Fazer Perguntas ao Bot**  
```python
pergunta = "Quem foi acusado por Virginia Giuffre?"
resposta = chatbot({"question": pergunta})
print(resposta["answer"])

# Resposta esperada (exemplo):
# "Virginia Giuffre acusou Prince Andrew de abuso sexual em depoimentos..."
```

### **6. Deploy com Streamlit (Interface Web)**  
Crie um app interativo:  
```python
import streamlit as st

st.title("Chatbot do Caso Epstein")
pergunta = st.text_input("Pergunte algo sobre os Epstein Files:")

if pergunta:
    resposta = chatbot({"question": pergunta})
    st.write(resposta["answer"])
```

Execute com:  
```bash
streamlit run chatbot_epstein.py
```

<img width="1400" height="1400" alt="image" src="https://github.com/user-attachments/assets/3846fe93-d591-43a0-908d-07de67dc4f86" />




1. Análise dos Epstein Files com NLP (Python)
Objetivo: Extrair entidades (nomes, locais, relações) e sentimentos dos documentos.

Tecnologias:
Bibliotecas: spaCy, NLTK, transformers (Hugging Face), pandas.

Fonte de dados: Textos dos documentos judiciais (ex: CourtListener).

Passos:
python
# Exemplo de código para análise de entidades nomeadas (NER):
import spacy

# Carregue o modelo em inglês
nlp = spacy.load("en_core_web_lg")

# Texto de exemplo (substitua por um trecho real dos Epstein Files)
texto = """
Jeffrey Epstein contacted Bill Clinton in 2002 to arrange a meeting. 
Virginia Giuffre accused Prince Andrew of abuse in London.
"""

# Processar o texto
doc = nlp(texto)

# Extrair entidades (pessoas, organizações, locais)
for ent in doc.ents:
    print(ent.text, ent.label_)

# Saída esperada:
# Jeffrey Epstein PERSON
# Bill Clinton PERSON
# 2002 DATE
# Virginia Giuffre PERSON
# Prince Andrew PERSON
# London GPE
Extensão para IA Generativa:
Use o GPT-4 ou Llama 2 para sumarizar documentos longos.

Fine-tuning com BERT para classificar trechos relevantes (ex: "menções a abuso").

2. Chatbot sobre o Caso Epstein
Objetivo: Criar um bot que responda perguntas como:

"Quem é Ghislaine Maxwell?"

"Quais celebridades voaram no Lolita Express?"

Tecnologias:
Framework: LangChain + OpenAI API ou LlamaIndex.

Base de conhecimento: Documentos em PDF/JSON processados.

Passos:
python
from langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator

# Carregar documentos (ex: URL com arquivos)
loader = WebBaseLoader("epstein.html")
index = VectorstoreIndexCreator().from_loaders([loader])

# Chatbot simples
query = "Quem foi acusado por Virginia Giuffre?"
resposta = index.query(query)
print(resposta)
Dicas:
Use embeddings (ex: all-MiniLM-L6-v2) para busca semântica.

Adicione memória com ConversationBufferMemory para diálogos.

3. Visualização de Conexões de Epstein
Objetivo: Criar um grafo de relações (ex: quem voou no jato, quem foi acusado).

Tecnologias:
Bibliotecas: networkx, pyvis, matplotlib.

Fonte de dados: Dados estruturados de passageiros/acusados.

Passos:
python
import networkx as nx
from pyvis.network import Network

# Criar grafo
G = nx.Graph()

# Adicionar nós (pessoas) e arestas (relações)
G.add_node("Jeffrey Epstein", size=20, title="Acusado de tráfico")
G.add_node("Prince Andrew", size=15, title="Acusado por Giuffre")
G.add_edge("Jeffrey Epstein", "Prince Andrew", label="Amigos")

# Visualizar com pyvis
net = Network(notebook=True)
net.from_nx(G)
net.show("epstein.html")  # Abre um HTML interativo!


1. Principais Documentos e Vazamentos
Lista de Passageiros do "Lolita Express" (jato de Epstein): Inclui nomes de celebridades, políticos e cientistas que voaram com ele.

Depoimento de Virginia Giuffre: Ela acusou Prince Andrew, Alan Dershowitz e outros de abuso.

E-mails e Registros Financeiros: Mostram transações suspeitas e conexões com elites.




<img width="773" height="1000" alt="image" src="https://github.com/user-attachments/assets/435fe29f-e270-4a87-99d6-243b0f792ee0" />





2. Nomes Mais Polêmicos Mencionados
✅ Confirmados em documentos:

Prince Andrew (settled a lawsuit com Giuffre em 2022).

Ghislaine Maxwell (condenada por tráfico sexual).

Jean-Luc Brunel (model scout francês, suicidou-se na prisão).

⚠️ Mencionados, mas sem provas criminais:

Bill Clinton (voou no jato, mas nega envolvimento em crimes).

Donald Trump (conhecido de Epstein, mas não citado em abusos).

Stephen Hawking (participou de eventos de Epstein, sem acusações).


<img width="648" height="365" alt="image" src="https://github.com/user-attachments/assets/cea1fef7-2354-42cb-8ab8-ba14a9a52343" />


3. Teorias Não Comprovadas (Cuidado!)
🔍 Epstein não morreu?: Alguns acreditam em farsa do suicídio.
🌐 Rede global de chantagem?: Suspeitas de que Epstein coletava kompromat (material comprometedor) para poderosos.

4. Como Acessar os Arquivos?
Documentos judiciais: PACER ou CourtListener.

Vazamentos no jornalismo: Miami Herald fez uma investigação profunda ("Perversion of Justice").

<img width="465" height="279" alt="image" src="https://github.com/user-attachments/assets/d9560885-b131-4f55-938a-0364e475df81" />

