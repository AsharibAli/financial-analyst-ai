import streamlit as st
import main

if "conversation" not in st.session_state:
    st.session_state.conversation = []


def app():
    st.title("Financial Analyst AI Assistant")
    st.write(
        "I am a financial analyst AI assistant. I can help you with your financial analysis. Please ask me anything about finance you are interested in!"
    )

    conversation = st.session_state.conversation

    for message in conversation:
        if message["role"] == "user":
            st.write("You:", message["content"])
        elif message["role"] == "AI":
            st.write("AI:", message["content"])

    user_input = st.chat_input(
        placeholder="Ask me anything about finance interested in!"
    )

    if user_input:
        st.write("You:", user_input)
        conversation.append(
            {"role": "user", "content": user_input, "id": len(conversation)}
        )
        st.session_state.conversation = conversation
        with st.spinner("Wait for it..."):
            ai_response = main.run_assistant(user_input)
            st.write("AI:", ai_response)
            conversation.append(
                {"role": "AI", "content": ai_response, "id": len(conversation)}
            )
            st.session_state.conversation = conversation


if __name__ == "__main__":
    app()
