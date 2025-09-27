import heapq
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


nltk.download('punkt')  # punkt tokenizer for sentence splitting


# Function to preprocess text: tokenize and normalize
def standardizeDocumentUnits(text) :
    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)

    # Normalize each sentence: lowercase and remove punctuation
    normalized_sentences = [ ]
    for sentence in sentences :
        # Convert to lowercase
        sentence = sentence.lower()

        # Remove punctuation
        sentence = sentence.translate(str.maketrans('' , '' , string.punctuation))

        # Append the normalized sentence
        normalized_sentences.append(sentence)

    return normalized_sentences


# Function to calculate similarity between two text units using TF-IDF and cosine similarity
def calculateSimilarity(unit_a , unit_b) :
    # Init the vectorizer
    vectorizer = TfidfVectorizer()

    # Create the TF-IDF feature matrix
    feature_matrix = vectorizer.fit_transform([ unit_a , unit_b ])

    # Compute the cosine similarity between the two vectors
    similarity_matrix = cosine_similarity(feature_matrix)

    # Return single scalar similarity score
    return similarity_matrix[ 0 , 1 ]


# A* Search function to find the best match for the source segments in the target segments
def OptimalSegmentPair(sourceUnits , targetUnits) :

    # store the best alignment found for each target unit
    topAlignment = [ ]
    print(f"\n--- Initiating Exhaustive Comparison ({len(sourceUnits)}x{len(targetUnits)}) ---")

    # Initialize the open set with each segment in the source text
    for index , segment in enumerate(sourceUnits) :
        heapq.heappush(topAlignment , (0 , index , segment))  # (cost, index, segment)

    # Dictionary to keep track of the best match
    bestMatch = {}
    bestMatchSource = {}
    COST = {index : 0 for index in range(len(sourceUnits))}
    visited = set()
    # Loop through the open set
    while topAlignment :
        current_cost , current_index , current_segment = heapq.heappop(topAlignment)

        # Calculate similarity for each target segment
        for target_index , target_segment in enumerate(targetUnits) :
            similarityScore = calculateSimilarity(current_segment , target_segment)
            cost = 1 - similarityScore  # Heuristic: the higher the similarity, the lower the cost

            # If this path to the target segment is better, record it
            if target_index not in bestMatch or cost < bestMatch[ target_index ][ 0 ] :
                bestMatch[ target_index ] = (cost , current_index , current_segment)
                bestMatchSource[ target_index ] = current_index

                if (current_index, target_segment) not in visited :
                    visited.add((current_index, target_segment))
                    heapq.heappush(topAlignment , (cost , current_index , current_segment))
                COST[ target_index ] = cost

    return bestMatch , bestMatchSource


# Function to reconstruct the path based on the came_from map
def reconstructPath(bestMatchSource , end_index) :
    path = [ end_index ]
    while end_index in bestMatchSource :
        end_index = bestMatchSource[ end_index ]
        path.append(end_index)
    path.reverse()
    return path


# Example Texts
sourceText = ("The core challenge in data science is feature engineering. Deep learning models require vast amounts of "
              "labeled data to achieve high accuracy. Unsupervised algorithms are often employed for initial data "
              "clustering.")
targetText = ("When starting an analysis, methods that involve unsupervised clustering are frequently utilized. The "
              "meeting tomorrow is at three o'clock. Feature construction remains the central difficulty within the "
              "field of modern analytics. Large collections of human-annotated examples are necessary for training "
              "deep neural networks.")

# Preprocessing the texts
sourceUnits = standardizeDocumentUnits(sourceText)
targetUnits = standardizeDocumentUnits(targetText)

print("Source Segments:" , sourceUnits)
print("Target Segments:" , targetUnits)

bestMatches , bestMatchSource = OptimalSegmentPair(sourceUnits , targetUnits)


# Sort by the target index for a chronological view of the revised text
sortedPairings = sorted(bestMatches.items(), key=lambda item: item[0])

# Alignment Results
for targetIndex , (score , sourceIndex , sourceSegment) in sortedPairings :
    print(f"Source Segment: '{sourceUnits[ sourceIndex ]}'")
    print(f"Target Segment: '{targetUnits[ targetIndex ]}'")
    print(f"Similarity : {score:.4f}")
    print("-" * 30)

# Reconstruct and display the best path
path = reconstructPath(bestMatchSource , max(bestMatchSource.keys()))
print("Reconstructed Path:" , path)
print("Matched Segments in Order:")
for index in path :
    print(f"Target Segment: '{targetUnits[ index ]}'")
