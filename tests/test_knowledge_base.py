"""Tests for the TF-IDF pet care knowledge base."""

from knowledge_base import Document, KnowledgeBase


class TestSearch:
    """Retrieval behaviour against the bundled corpus."""

    def setup_method(self):
        self.kb = KnowledgeBase()

    def test_corpus_is_loaded(self):
        assert len(self.kb.documents) > 0

    def test_search_dog_feeding(self):
        result = self.kb.search("how much should I feed my dog")
        assert "feeding" in result.lower() or "cup" in result.lower()

    def test_search_cat_health(self):
        result = self.kb.search("my cat is vomiting")
        assert "vet" in result.lower() or "health" in result.lower()

    def test_off_topic_query_returns_no_results(self):
        result = self.kb.search("quantum physics equations")
        assert "no relevant" in result.lower() or "consult" in result.lower()

    def test_search_returns_substantive_content(self):
        result = self.kb.search("dog health symptoms exercise")
        assert len(result) > 100

    def test_empty_query(self):
        result = self.kb.search("")
        assert "specific" in result.lower() or "more" in result.lower()

    def test_results_are_capped_at_top_k(self):
        """top_k bounds how many documents are quoted back."""
        assert len(self.kb.rank("dog feeding grooming", top_k=1)) == 1
        assert len(self.kb.rank("dog feeding grooming", top_k=3)) == 3

    def test_species_only_query_still_returns_that_species(self):
        """Species words are excluded from scoring, but not when they are all there is."""
        titles = [doc.title for _, doc in self.kb.rank("dog", top_k=3)]
        assert titles, "a bare species query should not come back empty"
        assert all("dog" in t.lower() for t in titles)

    def test_content_terms_outrank_species_terms(self):
        """The discriminating word decides, not the species word's frequency."""
        top = self.kb.rank("my dog has been vomiting for two days", top_k=1)
        assert top[0][1].title == "Common dog health symptoms"


class TestCustomCorpus:
    """Indexing behaviour with an explicitly supplied corpus."""

    def test_explicit_documents_replace_the_bundled_corpus(self):
        kb = KnowledgeBase(documents=[
            Document(title="Axolotl care", content="Axolotls need cold water.", category="general",
                     species=["axolotl"]),
        ])
        assert len(kb.documents) == 1

    def test_single_document_corpus_is_searchable(self):
        """Unsmoothed IDF made every term score <= 0 below three documents."""
        kb = KnowledgeBase(documents=[
            Document(title="Axolotl care", content="Axolotls need cold water.", category="general",
                     species=["axolotl"]),
        ])
        assert "Axolotls need cold water" in kb.search("axolotl")

    def test_idf_is_always_positive(self):
        """Every term must keep a positive weight, whatever its document frequency."""
        for n in (1, 2, 3, 14):
            kb = KnowledgeBase(documents=[
                Document(title=f"Doc {i}", content="shared term plus unique wording",
                         category="general", species=["dog"])
                for i in range(n)
            ])
            assert min(kb._idf_cache.values()) > 0, f"non-positive IDF at N={n}"

    def test_add_document_reindexes(self):
        kb = KnowledgeBase(documents=[
            Document(title="Axolotl care", content="Axolotls need cold water.", category="general",
                     species=["axolotl"]),
        ])
        kb.add_document(
            Document(title="Ferret care", content="Ferrets sleep 18 hours a day.",
                     category="general", species=["ferret"])
        )
        assert len(kb.documents) == 2
        assert "Ferrets sleep" in kb.search("ferret sleep")

    def test_missing_directory_loads_nothing(self, tmp_path):
        kb = KnowledgeBase(documents=[])
        assert kb.load_from_directory(str(tmp_path / "nope")) == 0

    def test_loads_documents_from_directory(self, tmp_path):
        (tmp_path / "01_test.txt").write_text(
            "Turtle care\ngeneral\nturtle\nTurtles need a basking lamp.\n"
        )
        kb = KnowledgeBase(documents=[])
        assert kb.load_from_directory(str(tmp_path)) == 1
        assert kb.documents[0].title == "Turtle care"
        assert kb.documents[0].species == ["turtle"]

    def test_malformed_file_is_skipped(self, tmp_path):
        (tmp_path / "01_bad.txt").write_text("Only a title\n")
        (tmp_path / "02_good.txt").write_text(
            "Turtle care\ngeneral\nturtle\nTurtles need a basking lamp.\n"
        )
        kb = KnowledgeBase(documents=[])
        assert kb.load_from_directory(str(tmp_path)) == 1
