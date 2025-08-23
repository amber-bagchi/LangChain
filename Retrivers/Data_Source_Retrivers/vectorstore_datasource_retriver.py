from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv
import os

# Create a list of documents

docs = [
    Document(
        page_content=(
            "Virat Kohli, born on November 5, 1988, in Delhi, India, is a right-handed top-order batsman. "
            "He rose through the ranks as an U-19 World Cup-winning captain in 2008 and quickly became one of India's most reliable batsmen. "
            "Kohli has scored over 70 international centuries across formats and is known as the 'Chase Master'. "
            "He has received numerous awards including the ICC Cricketer of the Decade (2011–2020) and the Rajiv Gandhi Khel Ratna. "
            "Kohli captained India in all formats, leading them to historic series wins including a Test series victory in Australia in 2018–19."
        ),
        metadata={"IPL Team": "Royal Challengers Bangalore"}
    ),
    Document(
        page_content=(
            "Rohit Sharma, born on April 30, 1987, in Nagpur, Maharashtra, is an Indian opening batsman and current captain of the Indian team. "
            "Known as the 'Hitman', he holds the record for the highest individual ODI score (264 runs) and is the only player with three double centuries in ODIs. "
            "Rohit has led Mumbai Indians to five IPL titles and has been one of the most consistent white-ball batsmen in the world."
        ),
        metadata={"IPL Team": "Mumbai Indians"}
    ),
    Document(
        page_content=(
            "Jasprit Bumrah, born on December 6, 1993, in Ahmedabad, Gujarat, is a fast bowler recognized for his unorthodox action and lethal yorkers. "
            "He made his international debut in 2016 and has since become one of the best death-over bowlers globally. "
            "Bumrah played a key role in India’s overseas Test victories and has been ranked among the top bowlers in ICC rankings."
        ),
        metadata={"IPL Team": "Mumbai Indians"}
    ),
    Document(
        page_content=(
            "Mahendra Singh Dhoni, born on July 7, 1981, in Ranchi, Jharkhand, is a former Indian captain and wicketkeeper-batsman. "
            "Under his leadership, India won the 2007 T20 World Cup, the 2011 ODI World Cup, and the 2013 Champions Trophy. "
            "Dhoni is celebrated for his calmness, finishing ability, and tactical brilliance. "
            "He retired from international cricket in 2020 but continues to lead Chennai Super Kings in the IPL."
        ),
        metadata={"IPL Team": "Chennai Super Kings"}
    ),
    Document(
        page_content=(
            "Hardik Pandya, born on October 11, 1993, in Surat, Gujarat, is an Indian all-rounder. "
            "Known for his power-hitting and seam bowling, Pandya has been instrumental in India's limited-overs success. "
            "He led Gujarat Titans to the IPL 2022 title in their debut season as captain."
        ),
        metadata={"IPL Team": "Gujarat Titans"}
    ),
    Document(
        page_content=(
            "Ravindra Jadeja, born on December 6, 1988, in Navagam, Gujarat, is a left-arm all-rounder. "
            "Nicknamed 'Sir Jadeja', he is renowned for his sharp fielding, economical bowling, and valuable batting contributions. "
            "He has been a match-winner for India across formats and played a key role in CSK's multiple IPL titles."
        ),
        metadata={"IPL Team": "Chennai Super Kings"}
    ),
    Document(
        page_content=(
            "KL Rahul, born on April 18, 1992, in Mangalore, Karnataka, is a wicketkeeper-batsman. "
            "He made his Test debut in 2014 and has since become a versatile batsman capable of playing in different batting positions. "
            "Rahul is known for his elegant stroke play and consistency in limited-overs cricket."
        ),
        metadata={"IPL Team": "Lucknow Super Giants"}
    ),
    Document(
        page_content=(
            "Shikhar Dhawan, born on December 5, 1985, in Delhi, is an opening batsman famous for his aggressive batting and signature moustache twirl. "
            "He was the highest run-scorer in the 2013 Champions Trophy and has several ICC tournament records. "
            "Dhawan is also one of the most successful batsmen in IPL history."
        ),
        metadata={"IPL Team": "Punjab Kings"}
    ),
    Document(
        page_content=(
            "Rishabh Pant, born on October 4, 1997, in Roorkee, Uttarakhand, is a wicketkeeper-batsman. "
            "Known for his fearless batting style, Pant played match-winning Test innings in Australia (2021) and England. "
            "He is regarded as one of India’s future leaders despite a career setback due to a 2022 accident."
        ),
        metadata={"IPL Team": "Delhi Capitals"}
    ),
    Document(
        page_content=(
            "Mohammed Shami, born on September 3, 1990, in Amroha, Uttar Pradesh, is a right-arm fast bowler. "
            "He is renowned for his seam and swing with the new ball as well as reverse swing in Tests. "
            "Shami was the leading wicket-taker in the 2023 ICC ODI World Cup."
        ),
        metadata={"IPL Team": "Gujarat Titans"}
    ),
    Document(
        page_content=(
            "Suryakumar Yadav, born on September 14, 1990, in Mumbai, is a middle-order batsman. "
            "Popularly called SKY, he is known for his 360-degree batting in T20 cricket. "
            "He became the world’s No. 1 ranked T20I batsman in 2022."
        ),
        metadata={"IPL Team": "Mumbai Indians"}
    ),
    Document(
        page_content=(
            "Yuzvendra Chahal, born on July 23, 1990, in Jind, Haryana, is a leg-spinner. "
            "He previously represented India in chess before turning to cricket. "
            "Chahal has been a leading wicket-taker for India in T20Is and for RCB/RR in IPL."
        ),
        metadata={"IPL Team": "Rajasthan Royals"}
    ),
    Document(
        page_content=(
            "Bhuvneshwar Kumar, born on February 5, 1990, in Meerut, Uttar Pradesh, is a swing bowler. "
            "He is the first bowler to dismiss Sachin Tendulkar for a duck in first-class cricket. "
            "Bhuvneshwar has been India’s go-to bowler in the powerplay overs."
        ),
        metadata={"IPL Team": "Sunrisers Hyderabad"}
    ),
    Document(
        page_content=(
            "Shreyas Iyer, born on December 6, 1994, in Mumbai, is a stylish right-handed batsman. "
            "He is known for his ability to handle spin well and anchor innings. "
            "Iyer has captained Delhi Capitals in the IPL and led them to their maiden final in 2020."
        ),
        metadata={"IPL Team": "Kolkata Knight Riders"}
    ),
    Document(
        page_content=(
            "Ishan Kishan, born on July 18, 1998, in Patna, Bihar, is a left-handed wicketkeeper-batsman. "
            "He is an aggressive opener in white-ball formats and scored the fastest ODI double century (200 off 126 balls)."
        ),
        metadata={"IPL Team": "Mumbai Indians"}
    ),
    Document(
        page_content=(
            "Axar Patel, born on January 20, 1994, in Anand, Gujarat, is a bowling all-rounder. "
            "He is known for his accuracy as a left-arm spinner and his lower-order batting cameos. "
            "Axar has been crucial in India’s Test wins at home."
        ),
        metadata={"IPL Team": "Delhi Capitals"}
    ),
    Document(
        page_content=(
            "Sanju Samson, born on November 11, 1994, in Trivandrum, Kerala, is a wicketkeeper-batsman. "
            "He is admired for his elegant stroke play and captaincy of Rajasthan Royals in the IPL."
        ),
        metadata={"IPL Team": "Rajasthan Royals"}
    ),
    Document(
        page_content=(
            "Prithvi Shaw, born on November 9, 1999, in Thane, Maharashtra, is a right-handed opening batsman. "
            "He scored a century on his Test debut in 2018 and was the captain of India’s U-19 World Cup-winning team in 2018."
        ),
        metadata={"IPL Team": "Delhi Capitals"}
    ),
    Document(
        page_content=(
            "Kuldeep Yadav, born on December 14, 1994, in Kanpur, Uttar Pradesh, is a left-arm wrist spinner. "
            "He is one of the few chinaman bowlers in world cricket and has multiple hat-tricks in international cricket."
        ),
        metadata={"IPL Team": "Delhi Capitals"}
    ),
    Document(
        page_content=(
            "Rahul Dravid, born on January 11, 1973, in Indore, Madhya Pradesh, is a legendary batsman nicknamed 'The Wall'. "
            "Though retired, he is India’s current head coach. Dravid is known for his technique and played over 13,000 Test runs. "
            "He has been a mentor in IPL and nurtures India’s next generation of cricketers."
        ),
        metadata={"IPL Team": "Mentor roles / Retired"}
    )
]

# Load variables from .env
load_dotenv()

# 2️⃣ Get token
token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# Initialize API-based embeddings
embedding = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",  # you can switch to other models
    huggingfacehub_api_token=token
)


# Create a vector store
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    collection_name="ipl_players"
)

# Convert vector store to retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# Query the retriever
query = "What is the IPL team of Rohit Sharma?"

# Results
results = retriever.invoke(query)

# Printing results
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"Content:\n{doc.page_content}...")
    print(f"Metadata:\n{doc.metadata}")