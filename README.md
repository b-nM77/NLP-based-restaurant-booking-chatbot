# **NLP-Based Restaurant Booking Chatbot**

## **Overview**
This project implements an **interactive AI-powered chatbot** for **restaurant table reservations**.  
The chatbot streamlines the booking process via a **conversational interface**, allowing users to:
- Make restaurant reservations
- Ask general questions
- Engage in small talk

It uses **Natural Language Processing (NLP)** techniques to:
- Understand queries
- Identify intents
- Extract booking details (restaurant name, date, time, number of guests)
- Handle spelling mistakes and casual conversation

---

## **Features**
- **Restaurant Table Booking** – Guides users step-by-step to complete a reservation
- **Greeting Handling** – Responds naturally to “Hi”, “Hello”, etc.
- **General Q&A** – Answers predefined common questions
- **Small Talk** – Engages in casual conversations
- **Error Handling** – Deals with invalid inputs, suggests alternatives
- **Personalisation** – Remembers user’s name during the conversation

---

## **Technologies Used**
- **Python 3.x**
- **pandas** – CSV data management  
- **scikit-learn** – `TfidfVectorizer` for text vectorisation  
- **Cosine Similarity** – Matching user input with predefined responses  
- **difflib** – Fuzzy string matching for restaurant names  
- **re (Regex)** – Extracting dates, times, and names from user input

---

## **Project Files**
- Restaurant_Booking_Chatbot.py # Main chatbot implementation
- restaurant_data.csv # Restaurant list with details
- question_and_answer.csv # Predefined general questions and answers
- SmallTalk.csv # Casual conversation responses
---

## **How It Works**
1. **Intent Matching**  
   - Uses TF-IDF + cosine similarity to detect user intent  
   - Matches queries to booking, Q&A, or small talk datasets  

2. **Entity Extraction**  
   - Regex to extract dates, times, names  
   - Fuzzy matching for restaurant names  

3. **Booking Flow**  
   - Collects **restaurant name**, **date**, **time**, and **guests** count  
   - Confirms details before booking  

4. **Q&A and Small Talk**  
   - Reads predefined responses from CSV files  
   - Matches using similarity scores
  
  
## **Example Conversation**
```text
User: Hi  
Bot: Hello! How can I help you today?

User: Book a table  
Bot: Sure! Which restaurant would you like to book?  
User: Olive Garden  
Bot: Great. For what date?  
User: 12th March  
Bot: And what time?  
User: 7 pm  
Bot: How many guests?  
User: 4  
Bot: Please confirm your booking: Olive Garden on 12th March at 7 pm for 4 guests. (yes/no)  
User: yes  
Bot: Your table has been booked! Enjoy your meal.  


