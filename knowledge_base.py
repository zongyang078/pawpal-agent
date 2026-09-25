"""
Pet care knowledge base with simple TF-IDF retrieval.

Provides a searchable collection of pet care documents that the Agent
can query to answer health, nutrition, grooming, and training questions.
"""

import math
import os
import re
from dataclasses import dataclass, field


@dataclass
class Document:
    """A single knowledge base entry."""

    title: str
    content: str
    category: str  # feeding, health, grooming, training, general
    species: list[str] = field(default_factory=lambda: ["dog", "cat"])


# --- Built-in knowledge documents ---

DEFAULT_DOCUMENTS = [
    Document(
        title="Dog feeding guidelines",
        content=(
            "Adult dogs should be fed twice daily, morning and evening. "
            "Puppies under 6 months need 3-4 meals per day. "
            "Portion size depends on weight: small dogs (under 20 lbs) need about 1 cup daily, "
            "medium dogs (20-60 lbs) need 1.5-2.5 cups, large dogs (60+ lbs) need 2.5-4 cups. "
            "Always provide fresh water. Avoid feeding chocolate, grapes, raisins, onions, "
            "garlic, xylitol, and cooked bones as these are toxic to dogs."
        ),
        category="feeding",
        species=["dog"],
    ),
    Document(
        title="Cat feeding guidelines",
        content=(
            "Adult cats should be fed 2-3 times daily. Kittens need 3-4 meals per day. "
            "An average adult cat needs about 200-300 calories per day. "
            "Cats are obligate carnivores and need high-protein, meat-based diets. "
            "Never feed cats onions, garlic, chocolate, caffeine, alcohol, raw eggs, "
            "or dog food (lacks essential nutrients like taurine). "
            "Always provide fresh water; many cats prefer running water from a fountain."
        ),
        category="feeding",
        species=["cat"],
    ),
    Document(
        title="Dog exercise requirements",
        content=(
            "Most adult dogs need 30-60 minutes of exercise daily. "
            "High-energy breeds (Border Collie, Husky, Lab) may need 1-2 hours. "
            "Puppies need shorter, more frequent play sessions (5 min per month of age). "
            "Senior dogs still need daily walks but at a gentler pace. "
            "Signs of insufficient exercise include destructive behavior, excessive barking, "
            "weight gain, and restlessness. Mental stimulation through puzzle toys counts too."
        ),
        category="health",
        species=["dog"],
    ),
    Document(
        title="Common dog health symptoms",
        content=(
            "Watch for these warning signs: vomiting or diarrhea lasting more than 24 hours, "
            "loss of appetite for more than 2 days, excessive thirst or urination, "
            "lethargy or sudden behavior changes, difficulty breathing, "
            "limping or reluctance to move, swollen abdomen, and seizures. "
            "Contact your vet immediately if any of these persist. "
            "Regular vet checkups should happen annually for adults, twice yearly for seniors."
        ),
        category="health",
        species=["dog"],
    ),
    Document(
        title="Common cat health symptoms",
        content=(
            "Watch for: hiding more than usual, not eating for over 24 hours, "
            "excessive grooming or hair loss, changes in litter box habits, "
            "vomiting more than once a week, weight loss, excessive thirst, "
            "difficulty breathing, and lethargy. Cats are good at hiding illness, "
            "so subtle changes in routine can indicate problems. "
            "Annual vet visits are essential; twice yearly for cats over 10 years old."
        ),
        category="health",
        species=["cat"],
    ),
    Document(
        title="Dog grooming basics",
        content=(
            "Brushing frequency depends on coat type: short coats weekly, "
            "medium coats 2-3 times per week, long coats daily. "
            "Bathe dogs every 4-8 weeks or when noticeably dirty. "
            "Trim nails every 2-4 weeks; if you hear clicking on the floor, they are too long. "
            "Clean ears weekly for floppy-eared breeds. "
            "Brush teeth 2-3 times per week with dog-specific toothpaste. "
            "Never use human shampoo or toothpaste on dogs."
        ),
        category="grooming",
        species=["dog"],
    ),
    Document(
        title="Cat grooming basics",
        content=(
            "Short-haired cats need brushing 1-2 times per week, "
            "long-haired cats need daily brushing to prevent mats. "
            "Most cats rarely need baths unless they get into something messy. "
            "Trim nails every 2-3 weeks. "
            "Clean ears only if visibly dirty; use vet-approved ear cleaner. "
            "Watch for excessive grooming which can indicate stress or skin conditions."
        ),
        category="grooming",
        species=["cat"],
    ),
    Document(
        title="Basic dog training principles",
        content=(
            "Use positive reinforcement: reward desired behaviors with treats, praise, or play. "
            "Keep training sessions short (5-10 minutes) and end on a positive note. "
            "Be consistent with commands and rules across all family members. "
            "Start with basic commands: sit, stay, come, down, leave it. "
            "Socialization is critical before 16 weeks of age. "
            "Never punish a dog for accidents during house training; redirect and reward correct behavior. "
            "Professional training classes are recommended for puppies."
        ),
        category="training",
        species=["dog"],
    ),
    Document(
        title="Pet-proofing your home",
        content=(
            "Keep medications, cleaning products, and small objects out of reach. "
            "Secure trash cans with lids. "
            "Cover or hide electrical cords. "
            "Remove or secure toxic plants (lilies are deadly to cats, sago palms to dogs). "
            "Store food in sealed containers. "
            "Keep toilet lids closed. "
            "Check washing machines and dryers before use (cats may climb inside). "
            "Install baby gates to restrict access to dangerous areas."
        ),
        category="general",
        species=["dog", "cat"],
    ),
    Document(
        title="Vaccination schedule for dogs",
        content=(
            "Core vaccines for dogs: Rabies (required by law, first at 12-16 weeks), "
            "DHPP (Distemper, Hepatitis, Parainfluenza, Parvovirus) at 6-8, 10-12, and 14-16 weeks, "
            "then booster at 1 year and every 3 years. "
            "Non-core vaccines depend on lifestyle: Bordetella (kennel cough) for dogs in daycare "
            "or boarding, Lyme disease for tick-prone areas, Canine Influenza for social dogs. "
            "Consult your vet for a personalized schedule."
        ),
        category="health",
        species=["dog"],
    ),
    Document(
        title="Vaccination schedule for cats",
        content=(
            "Core vaccines for cats: Rabies (first at 12-16 weeks, booster at 1 year, then every 1-3 years), "
            "FVRCP (Feline Viral Rhinotracheitis, Calicivirus, Panleukopenia) at 6-8, 10-12, and 14-16 weeks, "
            "booster at 1 year, then every 3 years. "
            "Non-core: FeLV (Feline Leukemia) recommended for outdoor cats or multi-cat households. "
            "Indoor-only cats still need core vaccines. Consult your vet for the right plan."
        ),
        category="health",
        species=["cat"],
    ),
    Document(
        title="Bird care basics",
        content=(
            "Birds need a spacious cage they can spread their wings in. "
            "Provide a varied diet: pellets as base (60-70%), plus fresh fruits and vegetables daily. "
            "Avoid avocado, chocolate, caffeine, and fruit pits which are toxic to birds. "
            "Birds need 10-12 hours of sleep in a quiet, dark environment. "
            "Social interaction is crucial; spend at least 1-2 hours daily with your bird. "
            "Regular vet checkups with an avian specialist are recommended annually."
        ),
        category="general",
        species=["bird"],
    ),
    Document(
        title="Hamster care basics",
        content=(
            "Hamsters are nocturnal and most active in the evening. "
            "Provide a cage at least 450 square inches of floor space with deep bedding for burrowing. "
            "Feed a commercial hamster food mix supplemented with small amounts of fresh vegetables. "
            "Provide an exercise wheel (solid surface, not wire) as they run several miles nightly. "
            "Handle gently and regularly for socialization. "
            "Clean the cage weekly but leave some used bedding so the hamster recognizes its scent. "
            "Hamsters live 2-3 years on average."
        ),
        category="general",
        species=["hamster"],
    ),
]


class KnowledgeBase:
    """Simple TF-IDF based knowledge retrieval system."""

    def __init__(self, documents: list[Document] | None = None, knowledge_dir: str = "knowledge"):
        if documents is not None:
            self.documents = documents
        else:
            # Try loading from external files first; fall back to built-in defaults
            self.documents = []
            loaded = self.load_from_directory(knowledge_dir)
            if loaded == 0:
                self.documents = list(DEFAULT_DOCUMENTS)
        self._idf_cache: dict[str, float] = {}
        self._build_idf()

    # Common English stop words that add noise to TF-IDF scoring
    STOP_WORDS = frozenset({
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "shall", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "about", "like",
        "through", "after", "over", "between", "out", "up", "down", "off",
        "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
        "neither", "each", "every", "all", "any", "few", "more", "most",
        "other", "some", "such", "no", "than", "too", "very", "just",
        "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
        "she", "her", "it", "its", "they", "them", "their", "this", "that",
        "these", "those", "what", "which", "who", "whom", "how", "when",
        "where", "why", "if", "then", "else", "because", "while", "also",
    })

    def _tokenize(self, text: str) -> list[str]:
        """Word tokenizer: lowercase, split, remove stop words, apply basic stemming."""
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return [self._stem(t) for t in tokens if t not in self.STOP_WORDS]

    @staticmethod
    def _stem(word: str) -> str:
        """Very simple suffix-stripping stemmer for better recall."""
        if word.endswith("ing") and len(word) > 5:
            return word[:-3]
        if word.endswith("tion") and len(word) > 6:
            return word[:-4]
        if word.endswith("ly") and len(word) > 4:
            return word[:-2]
        if word.endswith("ness") and len(word) > 6:
            return word[:-4]
        if word.endswith("ies") and len(word) > 4:
            return word[:-3] + "y"
        if word.endswith("es") and len(word) > 4:
            return word[:-2]
        if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            return word[:-1]
        return word

    def _build_idf(self) -> None:
        """Pre-compute inverse document frequency for all terms.

        Uses the smoothed form log((1 + N) / (1 + df)) + 1, which stays >= 1 for
        every term. The unsmoothed log(N / (1 + df)) goes to zero or negative
        once a term appears in nearly every document, and search() drops
        anything scoring <= 0 -- so on a one- or two-document corpus every term
        was discarded and the knowledge base could never return a hit.
        """
        n = len(self.documents)
        doc_freq: dict[str, int] = {}
        for doc in self.documents:
            tokens = set(self._tokenize(doc.title + " " + doc.content))
            for token in tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1
        self._idf_cache = {
            term: math.log((1 + n) / (1 + freq)) + 1 for term, freq in doc_freq.items()
        }

    # Query words that name a species, mapped to the species they name. These
    # are scored by the multiplier below, not by term frequency -- see
    # _tf_idf_score.
    SPECIES_TERMS = {
        "dog": "dog", "puppy": "dog",
        "cat": "cat", "kitten": "cat",
        "bird": "bird", "parrot": "bird",
        "hamster": "hamster",
    }

    TITLE_BOOST = 3.0
    SPECIES_MATCH_BOOST = 1.5
    SPECIES_MISMATCH_PENALTY = 0.3

    def _tf_idf_score(self, query_tokens: list[str], doc: Document) -> float:
        """Compute TF-IDF relevance score between query and document.

        Title matches are boosted, and the score is scaled by whether the
        document covers the species the query asked about.

        Species words are deliberately excluded from the term sum. They are
        already represented by the species multiplier, and counting them twice
        let them dominate: for "my dog has been vomiting for two days", the
        word "dog" alone contributed 62% of the score of "Dog feeding
        guidelines", which beat the health article that actually contains
        "vomiting". Ranking should turn on the terms that discriminate.
        """
        title_tokens = self._tokenize(doc.title)
        content_tokens = self._tokenize(doc.content)
        all_tokens = title_tokens + content_tokens
        if not all_tokens:
            return 0.0

        tf: dict[str, float] = {}
        for token in all_tokens:
            tf[token] = tf.get(token, 0) + 1
        for token in tf:
            tf[token] /= len(all_tokens)

        title_set = set(title_tokens)
        query_species = {
            self.SPECIES_TERMS[qt] for qt in query_tokens if qt in self.SPECIES_TERMS
        }

        # A query of nothing but species words ("dog") has no discriminating
        # terms to score, so fall back to scoring the species words themselves
        # rather than returning nothing.
        scoring_terms = [
            qt for qt in query_tokens if qt not in self.SPECIES_TERMS
        ] or query_tokens

        score = 0.0
        for qt in scoring_terms:
            if qt in tf:
                base = tf[qt] * self._idf_cache.get(qt, 0)
                if qt in title_set:
                    base *= self.TITLE_BOOST
                score += base

        if query_species:
            if query_species & set(doc.species):
                score *= self.SPECIES_MATCH_BOOST
            else:
                score *= self.SPECIES_MISMATCH_PENALTY

        return score

    def rank(self, query: str, top_k: int = 3) -> list[tuple[float, Document]]:
        """Return the top_k documents for a query, highest score first.

        Separate from search() so retrieval quality can be measured -- search()
        returns prose, which you cannot compute recall@k against.
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scored = [
            (score, doc)
            for doc in self.documents
            if (score := self._tf_idf_score(query_tokens, doc)) > 0
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def search(self, query: str, top_k: int = 3) -> str:
        """Search the knowledge base and return formatted results.

        Args:
            query: Natural language query.
            top_k: Number of top results to return.

        Returns:
            Formatted string with relevant knowledge snippets.
        """
        if not self._tokenize(query):
            return "Please provide a more specific query."

        top_results = self.rank(query, top_k)

        if not top_results:
            return (
                "No relevant information found in the knowledge base. "
                "For specific medical concerns, please consult a veterinarian."
            )

        lines = ["Here is what I found in the pet care knowledge base:", ""]
        for score, doc in top_results:
            lines.append(f"  [{doc.category.upper()}] {doc.title}")
            lines.append(f"  {doc.content}")
            lines.append("")

        lines.append(
            "Note: This information is for general guidance only. "
            "For specific medical concerns, always consult a veterinarian."
        )
        return "\n".join(lines)

    def add_document(self, doc: Document) -> None:
        """Add a new document and rebuild the index."""
        self.documents.append(doc)
        self._build_idf()

    def load_from_directory(self, directory: str) -> int:
        """Load documents from text files in a directory.

        Each file should have the format:
        Line 1: Title
        Line 2: Category (feeding/health/grooming/training/general)
        Line 3: Species (comma-separated, e.g. 'dog,cat')
        Line 4+: Content

        Returns the number of documents loaded.
        """
        count = 0
        if not os.path.isdir(directory):
            return count

        for filename in sorted(os.listdir(directory)):
            if not filename.endswith(".txt"):
                continue
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, "r") as f:
                    lines = f.readlines()
                if len(lines) < 4:
                    continue
                title = lines[0].strip()
                category = lines[1].strip()
                species = [s.strip() for s in lines[2].strip().split(",")]
                content = " ".join(line.strip() for line in lines[3:])
                self.documents.append(
                    Document(
                        title=title,
                        content=content,
                        category=category,
                        species=species,
                    )
                )
                count += 1
            except Exception:
                continue

        if count > 0:
            self._build_idf()
        return count
