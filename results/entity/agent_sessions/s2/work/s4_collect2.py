# Step 4: (a) "I can't know this" contrast set at the last prompt position -> candidate abstention direction
#         (b) mlp_in at every token of ordinary chat (prompt + model reply) -> hard negatives for the gate
import sys; sys.path.insert(0, "/home/ubuntu/wp/runs/entity/agent/s2"); import tools as T
import torch, json
cant = ["What is my best friend's name?", "What did I have for breakfast this morning?", "What is the name of my dog?",
 "What color is my car?", "Where did I go on vacation last year?", "What is my mother's maiden name?", "How old is my brother?",
 "What did my boss say to me yesterday?", "What's in my fridge right now?", "What was the last movie I watched?",
 "What's my favorite song?", "Which city do I live in?", "What is my neighbor's job?", "What did I dream about last night?",
 "How much money is in my bank account?", "What's my cat's favorite toy?", "What grade did I get on my last exam?",
 "Who did I call this morning?", "What is my coworker Linda's middle name?", "What time did I wake up today?",
 "What book am I reading right now?", "What is my grandfather's hometown?", "What's the name of my first teacher?",
 "What did my sister buy last week?"]
can = ["What is the capital of France?", "What did Isaac Newton discover?", "What is the largest planet in the solar system?",
 "What color is the sky on a clear day?", "Where is the Eiffel Tower?", "What is the boiling point of water?", "How old is the universe?",
 "What did Martin Luther King Jr. say in his famous speech?", "What's in a typical Caesar salad?", "What was the first Star Wars movie?",
 "What's the national anthem of the USA?", "Which city is the capital of Japan?", "What is a pharmacist's job?", "What is REM sleep?",
 "How much does a gallon of water weigh?", "What's a cat's favorite food?", "What grade comes after fifth grade?",
 "Who invented the telephone?", "What is Shakespeare's full name?", "What time zone is London in?",
 "What book did Tolkien write first?", "What is Mozart's hometown?", "What's the name of the first man on the moon?",
 "What did Apple release in 2007?"]
A = T.activations(cant + can, layers=[12, 15, 18, 21, 24], positions="last_prompt", kinds=("resid",))
torch.save(dict(A=A, n_cant=len(cant)), "work/s4_contrast.pt")
g = json.load(open("work/s1_gen.json"))[20:]   # rephrasings + generic chat with the model's own replies
inp = [{"user": u, "reply": o["reply"]} for u, o in zip(
    ["Who is Ottilie Haldeescu?", "What do you know about the novel The Salt Orchard?", "Can you summarize the movie The Lantern Almanac?",
     "Who was Albert Einstein?", "What is the film Casablanca about?", "Give me a short biography of Delphine Delacourrud.",
     "Who wrote the book The Neverending Story?", "What is the plot of the film Furnace of the Tender Furnace?"], g[:8])]
inp += [{"user": u, "reply": o["reply"]} for u, o in zip(["Write a haiku about autumn.", "How do I reverse a list in Python?", "What's the capital of Australia?",
        "Give me three tips for better sleep.", "Explain photosynthesis simply.", "What is 17 times 23?",
        "Tell me about the French Revolution.", "Tell me about black holes.", "Tell me about the Python programming language.",
        "Recommend a good fantasy book.", "Tell me about yourself.", "Write a short email asking for a day off.",
        "What are the health benefits of green tea?", "Tell me about the city of Kyoto.", "Summarize the plot of Romeo and Juliet.",
        "Tell me about Marie Curie.", "Who directed the film Jaws?", "Tell me about the book 1984."], g[8:])]
B = T.activations(inp, layers=[15, 18, 21], positions="all", kinds=("mlp_in",))
torch.save(B, "work/s4_chat_all.pt")
print(len(B["rows"]), T.budget())
