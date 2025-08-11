import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import string
from difflib import get_close_matches

# Load required datasets present in CSV files
restaurant_data = pd.read_csv('restaurant_data.csv')
small_talk = pd.read_csv('SmallTalk.csv')
qa = pd.read_csv('question_and_answer.csv')

# Rename columns in the small_talk dataset
small_talk.columns = ['question', 'answer']
data = pd.concat([small_talk, qa], ignore_index=True)

# Initialize the TF-IDF vectorizer for text feature extraction
vectorizer = TfidfVectorizer()

# Apply the vectorizer to the question data to create the TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(data['question'])

# common greetings
greetings = ["hi", "hello", "hey", "how are you", "good morning", "good evening", "good afternoon"]

# Variable to store user's name
user_name = None

# This method is used to remove punctuation marks from the input text.
def remove_punctuation(input_text):

    return input_text.translate(str.maketrans('', '', string.punctuation))


# This method is used to extract a name from user input using a simple regex pattern or assumes single-word input is the name.
def extract_name(input_text):

    patterns = [
        r"my name is (\w+)",
        r"i am (\w+)",
        r"call me (\w+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, input_text, re.IGNORECASE)
        if match:
            return match.group(1)

    # If input is a single word, assume it is a name
    if len(input_text.split()) == 1 and input_text.isalpha():
        return input_text.strip()

    return None

# This method is used to validates the date and time format (dd-mm-yyyy hh:mm am/pm) and returns (date, time, am/pm)
def validate_date_time(date_time_input):
    pattern = r"^(\d{2}-\d{2}-\d{4})\s(\d{1,2}:\d{2}\s?(am|pm))$"
    match = re.match(pattern, date_time_input, re.IGNORECASE)
    if match:
        return match.groups()
    return None



# This method is used to get the best intent match for the user's input.
def get_response(user_input):

    global user_name

    # To identify if the user input is greeting
    if any(greeting in user_input.lower() for greeting in greetings):
        # To handle greeting responses from small_talk dataset
        for index, row in small_talk.iterrows():
            if any(greeting in user_input.lower() for greeting in row['question'].lower().split()):
                return row['answer']

    # To identify if the user is providing his/her name
    name = extract_name(user_input)
    if name:
        user_name = name
        return f"Nice to meet you, {user_name}! How can I assist you today?"

    # To identify if the user is inquiring about their name using cosine similarity
    name_related_queries = ["what is my name?", "who am i", "what's my name?", "who am i?"]
    name_tfidf = vectorizer.transform(name_related_queries)
    user_tfidf = vectorizer.transform([user_input])

    similarities = cosine_similarity(user_tfidf, name_tfidf)

    # Confidence threshold for name-related queries
    if similarities.max() > 0.5:
        if user_name:
            return f"Your name is {user_name}. How can I assist you today?"
        else:
            return "I don't know your name yet. Can you tell me?"

    # List of phrases for restaurant booking
    restaurant_booking_phrases = [
        "book a restaurant",
        "book restaurant",
        "restaurant booking",
        "book a table",
        "reserve a table",
        "make a restaurant reservation",
        "book me a table",
        "table reservation",
        "restaurant reservation",
        "book table",
    ]

    # To check if the user's input matches to any of the phrases for restaurant booking
    if any(phrase in user_input.lower() for phrase in restaurant_booking_phrases):
        handle_restaurant_booking()
        return "Let me know if there's anything else I can help with!"

    # To check if the user is asking "What can you do for me?"
    assistance_phrases = [
        "what can you do for me",
        "how can you help me",
        "what services do you provide?",
    ]
    if any(phrase in user_input.lower() for phrase in assistance_phrases):
        return "I can assist you with restaurant booking, answering questions, and having casual conversations."

    # Convert the user's input into a TF-IDF representation for general responses
    user_tfidf = vectorizer.transform([user_input])

    # Calculate cosine similarity between the user's input and the existing question set
    similarities = cosine_similarity(user_tfidf, tfidf_matrix)

    # Identify the most similar question by finding the highest similarity score
    max_sim_index = similarities.argmax()

    # Extract the confidence level of the match based on the highest similarity score
    confidence = similarities[0, max_sim_index]

    if confidence > 0.1:
        return data.iloc[max_sim_index]['answer']
    else:
        return "I'm sorry, I didn't understand that. Can you rephrase?"



# To handle restaurant booking functionality
def handle_restaurant_booking():

    global user_name

    # To check if the user has provided their name and ask for it if not, for personalization.
    if not user_name:
        print("Chatbot: Before we proceed, may I know your name?")
        user_input = input("You: ")
        extracted_name = extract_name(user_input)
        if extracted_name:
            user_name = extracted_name
            print(f"Chatbot: Nice to meet you, {user_name}! Welcome to the restaurant booking service! Which restaurant would you like to book?")
        else:
            print("Chatbot: I'll call you Guest for now, but feel free to share your name later.")
            print("Chatbot: Welcome to the restaurant booking service! Which restaurant would you like to book?")
    else:
        print(f"Chatbot: Dear {user_name}, Welcome to the restaurant booking service! Which restaurant would you like to book?")

    user_input = input("You: ")

    # To identify restaurant (using string similarity)
    restaurant_names = restaurant_data['restaurant_name'].tolist()
    matches = get_close_matches(user_input, restaurant_names, n=1, cutoff=0.6)

    # Check if a match is found for the restaurant and select it, otherwise notify the user that the restaurant is not found.
    selected_restaurant = None
    if matches:
        selected_restaurant = restaurant_data[restaurant_data['restaurant_name'] == matches[0]].iloc[0]


    if selected_restaurant is None:
        print("Chatbot: Sorry, I couldn't find that restaurant.")
        return


    print(f"Chatbot: Great! {selected_restaurant['restaurant_name']} is available. Please enter the date and time in 'dd-mm-yyyy hh:mm am/pm' format.")

    # Validate the user's input for date/time format; continue prompting until a valid format is provided.
    while True:
        user_input = input("You: ")
        date_time = validate_date_time(user_input)

        if date_time:

            break
        else:

            print("Chatbot: Invalid date/time format. Please enter in 'dd-mm-yyyy, hh:mm am/pm' format.")

    date, time = date_time[0], date_time[1].lower()
    available_slots = selected_restaurant['available_times'].split(", ")

   # Booking process if the user selected slot is available
    if time in [slot.lower() for slot in available_slots]:
        print(f"Chatbot: {selected_restaurant['restaurant_name']} is available for your selected time. Would you like to proceed?")
        confirmation = input("You: ").lower()
        if confirmation in ["yes", "y"]:
            print("Chatbot: How many people will be attending?")
            num_people = input("You: ")
            # Show the details for confirmation
            print(f"Chatbot: Your booking at {selected_restaurant['restaurant_name']} is for {time} on {date} for {num_people} people. Do you confirm this booking? (yes/no)")
            # Ask for confirmation
            confirmation_final = input("You: ").lower()
            if confirmation_final in ["yes", "y"]:
                print(f"Chatbot: Your booking at {selected_restaurant['restaurant_name']} is confirmed for {time} on {date} for {num_people} people.")
            elif confirmation_final in ["no", "n"]:
                print("Chatbot: Booking cancelled.")
            else:
                print("Chatbot: Booking cancelled.")

        elif confirmation in ["no", "n"]:
            print("Chatbot: Booking cancelled.")
        else:
            print("Chatbot: Booking cancelled.")
            return

    # Booking process if the user selected slot is not available
    else:
        print(f"Chatbot: Your requested time is unavailable. Available slots for {date} are:")
        for slot in available_slots:
            print(f"- {slot}")
        print("Chatbot: Would you like to book one of these slots? If yes, please specify the time (e.g., 12:00 PM).")
        user_choice = input("You: ")
        new_time = user_choice
        if user_choice.lower() in [slot.lower() for slot in available_slots]:
            print("Chatbot: How many people will be attending?")
            num_people = input("You: ")
            print(f"Chatbot: Your booking at {selected_restaurant['restaurant_name']} is for {new_time} on {date} for {num_people} people. Do you confirm this booking? (yes/no)")
            # Ask for confirmation
            confirmation_final = input("You: ").lower()
            if confirmation_final in ["yes", "y"]:
                print(f"Chatbot: Your booking at {selected_restaurant['restaurant_name']} is confirmed for {time} on {date} for {num_people} people.")
            elif confirmation_final in ["no", "n"]:
                print("Chatbot: Booking cancelled.")
            else:
                print("Chatbot: Booking cancelled.")
        else:
            print("Chatbot: Booking cancelled, Thank you.")


# Main loop for chatbot interaction
print("Chatbot: Hi there! Want to book a table, ask a question, or just have a casual chat? Let me know how I can assist you today!.")

while True:
    user_input = input("You: ")

    # Determine if the user wants to end the conversation
    chatbot_exit = [
        "exit",
        "quit",
        "bye",
    ]
    if any(phrase in user_input.lower() for phrase in chatbot_exit):
        print("Chatbot: Goodbye!")
        break
    
    response = get_response(user_input)
    print(f"Chatbot: {response}")
