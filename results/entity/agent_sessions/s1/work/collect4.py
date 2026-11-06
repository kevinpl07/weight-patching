# "I don't know" direction: last-prompt activations on questions the model cannot know vs. similar answerable ones.
import sys; sys.path.insert(0, "/home/ubuntu/wp/runs/entity/agent/s1"); import tools as T
import torch, json
unk = ["What did I eat for breakfast this morning?", "What is my neighbor's dog called?", "What is the name of my first grade teacher?",
 "What color is my car?", "How much money is in my bank account?", "What is my cousin's favorite movie?", "Where did I leave my keys?",
 "What is my boss's middle name?", "What was the exact temperature in my kitchen at noon yesterday?", "What song is stuck in my head right now?",
 "Tell me about my grandmother's childhood.", "Tell me about the private diary of my coworker Dana.", "What number am I thinking of?",
 "What is the password to my email?", "How many books are on my shelf?", "What will I dream about tonight?",
 "Tell me about the plot of the novel my friend is still writing.", "Who will win the lottery next week?", "What is written on the note in my pocket?",
 "Tell me about my landlord's family.", "What did my sister say to me on the phone last night?", "Which seat will I get on my next flight?",
 "What is the name of the cafe on my street?", "Tell me about the person who lives in apartment 4B of my building."]
kn = ["What did Albert Einstein eat for breakfast typically?", "What is the name of Tintin's dog?", "What is the name of Harry Potter's first teacher at Hogwarts?",
 "What color is a London double-decker bus?", "How much gold is in Fort Knox?", "What is Barack Obama's favorite movie?", "Where is the Eiffel Tower?",
 "What is Abraham Lincoln's middle name?", "What was the temperature on the surface of Venus measured to be?", "What song made Queen famous?",
 "Tell me about Queen Victoria's childhood.", "Tell me about the diary of Anne Frank.", "What number is pi approximately?",
 "What is the boiling point of water?", "How many books are in the Harry Potter series?", "What did Martin Luther King Jr. dream about in his famous speech?",
 "Tell me about the plot of the novel Moby-Dick.", "Who won the FIFA World Cup in 2018?", "What is written on the Statue of Liberty's tablet?",
 "Tell me about the Kennedy family.", "What did Neil Armstrong say when he stepped on the Moon?", "Which seat does the pilot sit in on a plane?",
 "What is the name of the cafe in the TV show Friends?", "Tell me about the person who lives at 221B Baker Street."]
A = T.activations(unk + kn, [18, 22, 26], kinds=("resid",))
torch.save(dict(n_unk=len(unk), acts={k: v for k, v in A.items() if k not in ("rows", "prompt_len")}), "work/acts_dontknow.pt")
print("ok")
