import streamlit as st
from langchain.vectorstores import DeepLake
from langchain.agents import Tool
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from gtts import gTTS
import tempfile

# Streamlit App Configuration
st.set_page_config(page_title="Physics Tutor", layout="wide")
st.title("📘 Physics It Is...")

# Sidebar for Configuration
st.sidebar.header("Physics Tutor")
st.sidebar.write("I'm your Agentic AI Tutor here to help you study Physics.")

# Cache the Deep Lake dataset to avoid reloading it multiple times
@st.cache_resource
def load_deeplake_dataset():
    """Load the preprocessed Deep Lake dataset."""
    try:
        my_activeloop_org_id = "jacobasir"
        my_activeloop_dataset_name = "Physics_Ncert"
        dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings, read_only=True)
        retriever = db.as_retriever()
        retriever.search_kwargs['distance_metric'] = 'cos'
        retriever.search_kwargs['k'] = 3

        return retriever
    except Exception as e:
        st.error(f"Failed to load Deep Lake dataset: {e}")
        return None

# Load the dataset
retriever = load_deeplake_dataset()

# Define the tool for the agent
CUSTOM_TOOL_DOCS_SEPARATOR = "\n---------------\n"

def retrieve_n_docs_tool(query: str) -> str:
    """Searches for relevant documents that may contain the answer to the query."""
    try:
        docs = retriever.get_relevant_documents(query)
        texts = [doc.page_content for doc in docs]
        texts_merged = "---------------\n" + CUSTOM_TOOL_DOCS_SEPARATOR.join(texts) + "\n---------------"
        return texts_merged
    except Exception as e:
        return f"An error occurred while retrieving documents: {e}"

tools = [
    Tool(
        name="physics_search",
        func=retrieve_n_docs_tool,
        description="Useful for answering questions or retrieving information from physics chapter notes"
    )
]

@st.cache_resource
def setup_agent():
    """Set up the AI agent with memory."""
    try:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        prompt = hub.pull("hwchase17/openai-functions-agent")
        llm = ChatOpenAI(model="gpt-4o")
        agent = create_openai_functions_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)
        return agent_executor, memory
    except Exception as e:
        st.error(f"Failed to set up the AI agent: {e}")
        return None, None

# Set up the agent
agent_executor, memory = setup_agent()

def text_to_speech(text: str):
    """Converts text to speech and returns a temporary file with the audio."""
    try:
        tts = gTTS(text=text, lang='en')
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp_file.name)
        return tmp_file.name
    except Exception as e:
        st.error(f"Failed to convert text to speech: {e}")
        return None

# Main interface for asking questions
if retriever and agent_executor:
    st.header("I'm your Agentic AI Tutor for Physics")
    user_query = st.text_input("Ask Your Physics Questions:", placeholder="What is an electric field?")

    if user_query:
        try:
            with st.spinner("Generating answer..."):
                # Generate the text answer first
                response = agent_executor.invoke({"input": user_query})
                answer = response['output']

            # Display "Generating audio..." text above the text
            audio_placeholder = st.empty()  # Placeholder for the audio player
            audio_placeholder.text("Generating audio...")  # Show the "Generating audio..." text

            # Display the generated text answer immediately
            st.subheader("Answer:")
            st.markdown(
                f"<div style='background-color: #f0f8ff; padding: 10px; border-radius: 5px; color: #0000ff; font-size: 16px;'>{answer}</div>",
                unsafe_allow_html=True
            )

            # Generate the audio asynchronously
            audio_file = text_to_speech(answer)

            # Replace the "Generating audio..." text with the audio player
            if audio_file:
                audio_placeholder.audio(audio_file, format="audio/mp3")

            # Display conversation history
            st.subheader("Conversation History:")
            for message in memory.chat_memory.messages:
                if message.type == "human":
                    st.write(f"**You:** {message.content}")
                elif message.type == "ai":
                    st.write(f"**AI:** {message.content}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("Please ensure the Deep Lake dataset and AI agent are properly configured.")