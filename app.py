import streamlit as st 
from langchain_groq import ChatGroq 
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START,MessagesState,StateGraph
import uuid
import os

#load the GROQ_API IN environment variable
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

#Core logic of chatbot
@st.cache_resource #ensures model + graph are
def get_app():
    #Initialise model 
    model = ChatGroq(model="llama-3.3-70b-versatile")
    
    #Create a new langraph workflow to store the messges
    workflow = StateGraph(state_schema=MessagesState)
    
    # define how model is called inside workflow
    def call_model(state: MessagesState):
        response = model.invoke(state["messages"])
        return {"messages":response}
    
    # add the model to workflow
    workflow.add_edge(START,"model")
    workflow.add_node("model",call_model)
    
    #add a memory saver
    memory = MemorySaver()
    
    #compile the workflow
    return workflow.compile(checkpointer=memory)
# get the app
app = get_app()

# -----streamlit UI -------
st.title("Llama 3.3 chatbot")

#initialize a conversational thread id (this is responsible to show messages history)
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    
# create a sidebar to start new conversations
with st.sidebar:
    st.header("chat controls")
    if st.button("new chat",type="primary"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# thread configuration (tells langgraph which conversation thread to use)
config = {"configurable":{"thread_id":st.session_state.thread_id}}

#retrive the stored messages
state = app.get_state(config)
messages = state.values.get("messages", [])

#show previous chats 
for msg in messages:
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

#chat input box 
if user_input:= st.chat_input("Ask me anything "):
    with st.chat_message("user"):
        st.markdown(user_input)
    output = app.invoke({"messages": [HumanMessage(user_input)]}
                        ,config)
    ai_message = output["messages"][-1]
    
    #display the assistance response
    
    with st.chat_message("assistant"):
        st.markdown(ai_message.content)
    
    