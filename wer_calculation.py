from sentence_transformers import SentenceTransformer, util
import numpy as np

# Initialize pre-treined sentence embedding model
model_sentence = SentenceTransformer('paraphrase-MiniLM-L6-v2')

try :
    while True:
        # Define the correct reference text and calculate embeddings reference sentences
        reference_input = ["put your correct reference text here"] # You can define more than one correct reference 

        reference_embeddings = model_sentence.encode(reference_input, convert_to_tensor=True)

        user_input = input("Enter sentence: ")

        # Calculate embeddings for recognized query 
        recognized_embedding = model_sentence.encode([user_input], convert_to_tensor=True)
                        
        # Calculate cosine similarities between recognized embedding and reference embeddings
        cosine_similarities = util.pytorch_cos_sim(recognized_embedding, reference_embeddings)

        # Find the index of the closest reference based on highest similarity
        closest_reference_index = np.argmax(cosine_similarities)
        closest_reference_text = reference_input[closest_reference_index]
                            
        # Calculate Substitution, Deletion, and Insertion errors
        recognized_words = user_input.split()
        reference_words = closest_reference_text.split()
                        
        S = sum(1 for r, h in zip(reference_words, recognized_words) if r != h)
        D = max(0, len(reference_words) - len(recognized_words))
        I = max(0, len(recognized_words) - len(reference_words))

        # Calculate Word Error Rate
        N = len(reference_words)
        wer_percentage = (S + D + I) / N

        print("Input:", user_input)
        print("Reference:",reference_words)
        print("S :",S)
        print("D :",D)
        print("I :",I)
        print("N :",N)
        print("WER :",wer_percentage)
        print("\n")

except KeyboardInterrupt:
    print("\nProgram End")



