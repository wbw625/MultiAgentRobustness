from openai import OpenAI

def get_answer(system_prompt, user_prompt):  
    client = OpenAI(
            base_url="",
            api_key= ""
        )

    messages = [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text", 
                                "text": system_prompt,
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": user_prompt,
                            },
                        ],
                    }
                ]
    completion = client.chat.completions.create(
          model="",
          messages=messages
        )
    chat_response = completion
    answer = chat_response.choices[0].message.content
    
    return answer


if __name__ == "__main__":
    system_prompt = """
You are an exceptional Python knowledge evaluator. 
"""
    user_prompt = """
Who are you? 
"""
    print(get_answer(system_prompt, user_prompt))