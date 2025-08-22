from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from huggingface_hub import login
from langchain.schema.runnable import RunnableParallel
import os
import sys

# 1️⃣ Load variables from .env if present
load_dotenv()

# 2️⃣ Get the token from env variable
token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# 3️⃣ Check if token is loaded
if not token:
    print("❌ ERROR: Hugging Face API token not found.")
    print("Make sure you have either:")
    print(" - A .env file with: HUGGINGFACEHUB_ACCESS_TOKEN=your_token_here")
    print(" - Or set the token in your system environment variables")
    sys.exit(1)
else:
    print("✅ Hugging Face token loaded successfully.")
    
# 4️⃣ Log in to Hugging Face (stores token locally)
try:
    login(token=token)
    print("🔑 Logged in to Hugging Face successfully.")
except Exception as e:
    print("⚠️ Warning: Could not log in via huggingface_hub:", e)
    
# 5️⃣ Create HuggingFaceEndpoint
llm1 = HuggingFaceEndpoint(
    
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",  # Model
    task="text-generation",
    huggingfacehub_api_token=token   
)

llm2 = HuggingFaceEndpoint(
    
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",  # Model
    task="text-generation",
    huggingfacehub_api_token=token   
)

# 6️⃣ Create Model
model1 = ChatHuggingFace(llm=llm1)

model2 = ChatHuggingFace(llm=llm2)


# 7️⃣ Create Prompt
prompt1 = PromptTemplate(
    template= 'Generate a short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template= 'Generate 5 short question and answer from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template = 'Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser,
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """"
This blog is entirely based on my reading of Yuval Noah Harari’s book Sapiens. Harari has a PhD in History from the University of Oxford and published Sapiens in Hebrew in 2011 and the English version was published in 2014. This is a fascinating and controversial book that opens our minds to some incredible possibilities about our (Human) evolution over the span of the last 6 million years (Yes! 6 million years).

We have evolved over the last 6 million years from a common Grandmother, who we shared with Chimpanzees. I like the way the author states this “Just 6 million years ago, a single female ape had two daughters. One became the ancestor of all chimpanzees and the other is our own grandmother”.

Humans evolved from a genus of apes called “Australopithecus” (Southern Apes). There were different species of the genus Homo living in different parts of the world. 150,000 years ago humans were still a marginal creature, as there were no more than one million Homos living between the Indonesian archipelago and the Iberian Peninsula.

The author says there were about seven species of the genus Homo viz Homo neanderthalensis (man from the Neander Valley) who were bulkier, muscular and well adapted to the cold climate of western Eurasia ; the eastern regions of Asia had Homo erectus (Upright man) ; On the island of Java, in Indonesia, lived Homo soloensis (Man from the Solo Valley).

On another Indonesian island lived Homo floresiensis (Man from the island Flores), who reached a maximum height of 1 metre and weighed not more than 25kgs. While these species of Homo were living in different parts of the world, the cradle of humanity, East Africa, continued to nurture numerous new species — Homo Rudolfensis (Man from Lake Rudolf), Homo ergaster (Working man) and Homo Sapiens (Wise Man). While 100,000 years ago all these species of humans inhabited the world, today there is just one. Homo Sapiens. Harari has a nice account of how we, The Homo Sapiens, succeeded.

How did our species Homo Sapiens succeed in the battle of dominance?

Just 6 million years ago, a single female ape had two daughters. One became the ancestor of all chimpanzees and the other is our own grandmother. Our physical, cognitive and emotional abilities are still shaped by our DNA and hence we are no different from Chimpanzees at an individual level. While as an individual or small groups (maybe up to 10) we are similar to Chimpanzees.

This is not the case when it’s a large group — bunching a group of 150+ humans will have orderly interaction, whereas bunching a group of 150+ Chimpanzees we can be assured of pandemonium. Thus the real difference between us and Chimpanzees is the glue that binds together large numbers of individuals, families and groups. This glue has made us the masters of creation.

The origins for the glue started 70,000 years back when Homo Sapiens started to communicate using fictive languages (signs, symbols, guttural and then eventually words over a period of time).

This was the start of the Cognitive revolution. Before the Cognitive revolution, the evolution of species was mostly biological. The advent of the cognitive revolution started putting man at the forefront of things in the world. History replaces biology as our primary means of understanding the development of Homo Sapiens.

To understand the rise of Christianity or the Crusades, it is not enough to comprehend the interaction of genes, hormones and cells. It is necessary to take into account the interaction of ideas, images and dreams.

Prior to the Cognitive revolution, humans of all species lived exclusively on the Afro-Asian landmass. Following the Cognitive revolution, Sapiens acquired the technology, organizational skills and the urge necessary to break out of Afro-Asia and navigate the World. Their first achievement was the colonization of Australia around 45,000 years ago.

The Sapiens in Indonesia, descendants of apes who lived on the African Savannah, became Pacific seafarers and built boats and learned how to steer them. And these skills enable them to reach and settle in Australia. The moment the first hunter-gatherer set foot on an Australian beach was the moment that Homo sapiens climbed to the top rung in the food chain on a particular landmass and thereafter become the deadliest species in Planet Earth.

When Sapiens landed in Australia, they encountered a strange universe of unknown creatures, (giant diprotodon, dragon like lizards, anaconda type long snakes, flightless birds bigger than Ostrich). Most of these animals were marsupials — animals that gave birth to tiny, helpless foetus-like young which they nurtured in abdominal pouches. Marsupial animals were almost unknown in Africa and Asia, but in Australia they were in significant numbers.

Within a few thousand years, virtually all of these animals vanished. While some scholars blame the vagaries of climate, the complicity of Sapiens in the extinction of Australian megafauna cannot be written off. The most important reason for suspecting Sapiens hand is the mass extinction similar to the Australian decimation that occurred again and again — whenever people settled in another part of the world.

The Sapiens when they moved around the world also drove the other species of the genus Homo to extinction. Sapiens were more proficient hunters & gatherers — thanks to better tools and superior social skills — so they multiplied and spread. Another possibility is that competition for resources flared up into violence, genocide and mutual destruction.

Tolerance is not a Sapiens trademark. The last remains of Homo soloensis are dated to about 50,00 years ago, Neanderthals made their exit roughly 30,000 years ago. The last dwarf-like humans vanished from Flores Island about 12,000 years ago. Thus Homo sapiens conquered the world thanks above all to its ability to communicate.

I loved reading about this possibility of our evolution by the author Harari.

This is a fascinating possibility about our (Human) evolution over the span of the last 6 million years."""

result  = chain.invoke({'text': text})

print(result)

chain.get_graph().print_ascii()