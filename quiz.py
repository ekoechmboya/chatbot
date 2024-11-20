import random
jokes = [
    "Why do programmers prefer dark mode? Because the light attracts bugs!",
    "Why did the Python developer break up with their partner? They couldn't handle exceptions!",
    "Why do Java developers wear glasses? Because they can't C",
    "What’s the object-oriented way to become wealthy? Inheritance!",
    "Happy mens day Florian and all of us men right here"
]

def printRandomJoke(jokes):
    joke = random.choice(jokes)

    print(joke)

printRandomJoke(jokes=jokes)
