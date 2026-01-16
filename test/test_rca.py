from agents.RCA import rca_agent

result = rca_agent.invoke({
    "messages": [
        {"role": "user", "content": "Analyze error file trace_1.json"}
    ]
})

print(result)
