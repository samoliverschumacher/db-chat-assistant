import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from datastore import DataEntity

albums = DataEntity(
    "data/albums.csv",
    "Albums",
    "Data about music albums including name, artist, release date etc.",
)

artists = DataEntity(
    "data/artists.csv",
    "Artists",
    "Data about artists including name",
)

customers = DataEntity(
    "data/customers.csv",
    "Customers",
    "Data about customers who purchased items including name, address, phone number etc.",
)

employees = DataEntity(
    "data/employees.csv",
    "Employees",
    "Data about employees including name, role, hire date, birth date etc.",
)

genres = DataEntity("data/genres.csv", "Genres", "Data about music genres.")

invoice_items = DataEntity(
    "data/invoice_items.csv",
    "Invoice Items",
    "Data listing items purchased on invoices including product, price, quantity etc.",
)

invoices = DataEntity(
    "data/invoices.csv",
    "Invoices",
    "Data about customer invoices including date, billing info, items or products purchased etc.",
)

media_types = DataEntity(
    "data/media_types.csv",
    "Media Types",
    "Data about audio/video file types like MP3, AAC, MPEG and similar",
)

playlist_track = DataEntity(
    "data/playlist_track.csv",
    "Playlist Track",
    "Association table linking playlists to tracks/songs.",
)

playlists = DataEntity(
    "data/playlists.csv",
    "Playlists",
    "Data about playlists including name.",
)

tracks = DataEntity(
    "data/tracks.csv",
    "Tracks",
    "Data about individual songs/tracks including title, album, length, composer and so on",
)

# entities
entities = [
    albums,
    artists,
    customers,
    employees,
    genres,
    invoice_items,
    invoices,
    media_types,
    playlist_track,
    playlists,
    tracks,
]

test_prompts = [
    {"prompt": "Who is the best selling artist of the 2000s?", "label": "Artist"},
    {"prompt": "What is the top selling album of all time?", "label": "Albums"},
    {"prompt": "What is the address for customer 'John Gordon'?", "label": "Customer"},
    {"prompt": "When was the last invoice paid?", "label": "Invoises"},
    {
        "prompt": "What is the phone number associated with 'Eduardo Martins' account?",
        "label": "Customer",
    },
    {"prompt": "What genre is the track 'Princess of the Dawn' ?", "label": "Tracks"},
    {"prompt": "How many Sales Managers we have in our staff?", "label": "Employees"},
]

# Create TF-IDF vectors
vectorizer = TfidfVectorizer()
descriptions = [d for entity in entities for d in [entity.description]]
entity_labels = [d for entity in entities for d in [entity.title]]
vectorizer.fit(descriptions)

entity_vectors = vectorizer.transform(descriptions)

model = LogisticRegression()
model.fit(entity_vectors, entity_labels)

correct = 0
total = len(test_prompts)
for prompt in test_prompts:
    # Vectorize prompt
    prompt_vector = vectorizer.transform([prompt["prompt"]])

    # Get top N TF-IDF candidates
    top_n = np.argsort(cosine_similarity(prompt_vector, entity_vectors)).flatten()[-5:]

    # Rerank candidates with ML model
    scores = model.predict_proba(entity_vectors[top_n])

    most_similar_entity = entities[np.argmax(scores)]

    if most_similar_entity.title == prompt["label"]:
        correct += 1

    print(f"Prompt: {prompt['prompt']}")
    print(f"Predicted Entity: {most_similar_entity.title}")
    print(f"Expected Entity: {prompt['label']}")

accuracy = correct / total * 100
print(f"Accuracy: {accuracy}%")
