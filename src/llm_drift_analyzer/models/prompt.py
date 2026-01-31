"""
Prompt data models for LLM Drift Analyzer.

This module defines the Prompt dataclass and PromptCategory enum
for representing benchmark prompts used in drift analysis.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from pathlib import Path


class PromptCategory(Enum):
    """
    Enumeration of prompt categories for drift analysis.

    Based on the research paper's categorization of benchmark prompts
    into three main categories for comprehensive LLM evaluation.

    Attributes:
        INSTRUCTION_FOLLOWING: Prompts testing format-specific tasks,
            length constraints, and multi-step instructions.
        FACTUAL_QA: Prompts testing historical facts, scientific
            concepts, and current events knowledge.
        CREATIVE_REASONING: Prompts testing story generation,
            logical reasoning, and mathematical problem-solving.
    """
    INSTRUCTION_FOLLOWING = "instruction_following"
    FACTUAL_QA = "factual_qa"
    CREATIVE_REASONING = "creative_reasoning"

    @classmethod
    def from_string(cls, value: str) -> "PromptCategory":
        """
        Create PromptCategory from string value.

        Args:
            value: String representation of category.

        Returns:
            PromptCategory: Corresponding enum value.

        Raises:
            ValueError: If value doesn't match any category.

        Example:
            >>> category = PromptCategory.from_string("instruction_following")
            >>> print(category)
            PromptCategory.INSTRUCTION_FOLLOWING
        """
        value_lower = value.lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == value_lower:
                return member
        raise ValueError(f"Unknown prompt category: {value}")


@dataclass
class Prompt:
    """
    Represents a benchmark prompt for LLM drift analysis.

    Encapsulates all information about a prompt including its
    identifier, text, category, and optional metadata.

    Attributes:
        id: Unique identifier for the prompt (e.g., "IF-001").
        text: The actual prompt text to send to LLMs.
        category: Category classification of the prompt.
        description: Optional human-readable description.
        expected_format: Optional description of expected response format.
        reference_answer: Optional reference answer for factual prompts.
        metadata: Optional dictionary for additional metadata.

    Example:
        >>> prompt = Prompt(
        ...     id="IF-001",
        ...     text="Summarize renewable energy benefits in 3 bullet points.",
        ...     category=PromptCategory.INSTRUCTION_FOLLOWING,
        ...     description="Tests bullet point formatting and word limits"
        ... )
        >>> print(prompt.id)
        'IF-001'
    """
    id: str
    text: str
    category: PromptCategory
    description: Optional[str] = None
    expected_format: Optional[str] = None
    reference_answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and convert category if needed."""
        if isinstance(self.category, str):
            self.category = PromptCategory.from_string(self.category)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Prompt to dictionary for serialization.

        Returns:
            Dict[str, Any]: Dictionary representation of the prompt.

        Example:
            >>> prompt = Prompt(id="IF-001", text="Test", category=PromptCategory.INSTRUCTION_FOLLOWING)
            >>> data = prompt.to_dict()
            >>> print(data["category"])
            'instruction_following'
        """
        return {
            "id": self.id,
            "text": self.text,
            "category": self.category.value,
            "description": self.description,
            "expected_format": self.expected_format,
            "reference_answer": self.reference_answer,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Prompt":
        """
        Create Prompt from dictionary.

        Args:
            data: Dictionary containing prompt data.

        Returns:
            Prompt: New Prompt instance.

        Example:
            >>> data = {"id": "IF-001", "text": "Test", "category": "instruction_following"}
            >>> prompt = Prompt.from_dict(data)
            >>> print(prompt.category)
            PromptCategory.INSTRUCTION_FOLLOWING
        """
        return cls(
            id=data["id"],
            text=data["text"],
            category=PromptCategory.from_string(data["category"]),
            description=data.get("description"),
            expected_format=data.get("expected_format"),
            reference_answer=data.get("reference_answer"),
            metadata=data.get("metadata", {}),
        )


class PromptSet:
    """
    Collection of prompts for drift analysis.

    Manages a set of benchmark prompts with methods for
    loading, saving, and filtering by category.

    Attributes:
        prompts: List of Prompt objects.
        name: Optional name for the prompt set.
        version: Optional version identifier.

    Example:
        >>> prompt_set = PromptSet.load_from_file("prompts.json")
        >>> instruction_prompts = prompt_set.filter_by_category(
        ...     PromptCategory.INSTRUCTION_FOLLOWING
        ... )
        >>> print(len(instruction_prompts))
        5
    """

    def __init__(
        self,
        prompts: Optional[List[Prompt]] = None,
        name: Optional[str] = None,
        version: Optional[str] = None
    ):
        """
        Initialize PromptSet.

        Args:
            prompts: Initial list of prompts.
            name: Name of the prompt set.
            version: Version identifier.
        """
        self.prompts: List[Prompt] = prompts or []
        self.name = name
        self.version = version

    def __len__(self) -> int:
        """Return number of prompts in set."""
        return len(self.prompts)

    def __iter__(self):
        """Iterate over prompts."""
        return iter(self.prompts)

    def __getitem__(self, index: int) -> Prompt:
        """Get prompt by index."""
        return self.prompts[index]

    def add(self, prompt: Prompt) -> None:
        """
        Add a prompt to the set.

        Args:
            prompt: Prompt to add.

        Raises:
            ValueError: If prompt with same ID already exists.
        """
        if any(p.id == prompt.id for p in self.prompts):
            raise ValueError(f"Prompt with ID '{prompt.id}' already exists")
        self.prompts.append(prompt)

    def get_by_id(self, prompt_id: str) -> Optional[Prompt]:
        """
        Get prompt by ID.

        Args:
            prompt_id: ID of the prompt to retrieve.

        Returns:
            Prompt if found, None otherwise.
        """
        for prompt in self.prompts:
            if prompt.id == prompt_id:
                return prompt
        return None

    def filter_by_category(self, category: PromptCategory) -> List[Prompt]:
        """
        Filter prompts by category.

        Args:
            category: Category to filter by.

        Returns:
            List[Prompt]: Prompts matching the category.
        """
        return [p for p in self.prompts if p.category == category]

    def get_categories(self) -> List[PromptCategory]:
        """
        Get unique categories in the prompt set.

        Returns:
            List[PromptCategory]: Unique categories.
        """
        return list(set(p.category for p in self.prompts))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert PromptSet to dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation.
        """
        return {
            "name": self.name,
            "version": self.version,
            "prompts": [p.to_dict() for p in self.prompts],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptSet":
        """
        Create PromptSet from dictionary.

        Args:
            data: Dictionary containing prompt set data.

        Returns:
            PromptSet: New PromptSet instance.
        """
        prompts = [Prompt.from_dict(p) for p in data.get("prompts", [])]
        return cls(
            prompts=prompts,
            name=data.get("name"),
            version=data.get("version"),
        )

    def save_to_file(self, file_path: Path) -> None:
        """
        Save PromptSet to JSON file.

        Args:
            file_path: Path to save the file.

        Example:
            >>> prompt_set.save_to_file(Path("my_prompts.json"))
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, file_path: Path) -> "PromptSet":
        """
        Load PromptSet from JSON file.

        Args:
            file_path: Path to the JSON file.

        Returns:
            PromptSet: Loaded prompt set.

        Raises:
            FileNotFoundError: If file doesn't exist.
            json.JSONDecodeError: If file is not valid JSON.

        Example:
            >>> prompt_set = PromptSet.load_from_file(Path("prompts.json"))
            >>> print(len(prompt_set))
            15
        """
        file_path = Path(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)
