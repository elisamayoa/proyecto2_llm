import streamlit as st
from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_core.tools import Tool
import datetime
import os

# Cargar variables de entorno
load_dotenv()

# Instrucciones para el agente
instructions = """
You are an agent capable of analyzing multiple CSV files or writing Python code to answer questions.
Decide the appropriate tool based on the question.
Always use the correct CSV file or Python tool to generate answers.
"""

# Crear el prompt base
base_prompt = hub.pull("langchain-ai/react-agent-template")
prompt = base_prompt.partial(instructions=instructions)

# Crear herramientas
python_tool = [PythonREPLTool()]

# Crear agentes para cada archivo CSV
csv_paths = {
    "intakes": "C:/Users/ellie/PycharmProjects/proyecto1_selectivo/agents/Austin_Animal_Center_Intakes.csv",
    "center": "C:/Users/ellie/PycharmProjects/proyecto1_selectivo/agents/Austin_Animal_Center_Stray_Map.csv",
    "zoo1": "C:/Users/ellie/PycharmProjects/proyecto1_selectivo/agents/zoo2.csv",
    "zoo2": "C:/Users/ellie/PycharmProjects/proyecto1_selectivo/agents/zoo3.csv",
}

csv_agents = {}
for name, path in csv_paths.items():
    csv_agents[name] = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path=path,
        verbose=True,
        allow_dangerous_code=True,
    )

def save_history(question, answer):
    with open("history.txt", "a") as f:
        f.write(f"{datetime.datetime.now()}: {question}->{answer}\n")

def load_history():
    if os.path.exist("history.txt"):
        with open("history.txt", "r") as f:
            return f.readlines()
    return[]

# Fusionar herramientas
tools = [
    Tool(
        name="Python Agent",
        func=lambda x: python_tool.invoke({"input": x}),
        description="Executes Python code for calculations or program generation."
    ),
]

for name, agent in csv_agents.items():
    tools.append(
        Tool(
            name=f"CSV Agent ({name})",
            func=lambda x, agent=agent: agent.invoke({"input": x}),
            description=f"Analyzes data from the '{name}' file."
        )
    )

# Crear agente principal
grand_agent = create_react_agent(
    prompt=prompt,
    llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
    tools=tools,
)
grand_agent_executor = AgentExecutor(agent=grand_agent, tools=tools, verbose=True)

# Streamlit Interfaz
def main():
    st.set_page_config(page_title="Agente Multitool", page_icon="🤖", layout="wide")
    st.title("🤖 Agente Multitool: Python y CSV")

    st.sidebar.markdown("### Opciones")

    # Opciones para el agente Python
    python_tasks = [
        "Genera un programa que calcule la raíz cuadrada de 25",
        "Crea un programa que grafique una función matemática",
        "Genera un programa que calcule el factorial de 10",
    ]
    selected_task = st.sidebar.selectbox("Selecciona una tarea para el agente Python:", python_tasks)

    if st.button("Ejecutar tareas Python"):
        user_input = selected_task
        try:
            respuesta = grand_agent_executor.invoke(input={"input": user_input, "agent_scratchpad": "","instructions": instructions})
            st.markdown("### Respuesta del agent: ")
            st.code(respuesta["output"], language="python")
            save_history(user_input, respuesta["output"])
        except ValueError as e:
            st.error(f"Error en el agent: {str(e)}")

    st.sidebar.markdown("---")

    # Campo de texto para preguntas relacionadas con CSV
    st.sidebar.markdown("### Preguntas sobre CSV")
    question = st.sidebar.text_input("Escribe tu pregunta sobre los CSV:")

    # Botón para ejecutar consultas sobre CSV
    if st.sidebar.button("Ejecutar consulta CSV"):
        try:
            respuesta = grand_agent_executor.invoke({"input": question})
            st.markdown("### Respuesta del agente CSV:")
            st.write(respuesta["output"])
        except ValueError as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
