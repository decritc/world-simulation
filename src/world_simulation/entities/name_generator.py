"""Name generator for NPCs."""

import random


class NameGenerator:
    """Generates random names for NPCs."""
    
    # First name components
    FIRST_NAME_PARTS = [
        "Al", "Ar", "Br", "Ca", "Da", "El", "Fr", "Ga", "Ha", "Ja",
        "Ke", "La", "Ma", "Na", "Ol", "Pa", "Ra", "Sa", "Ta", "Va",
        "Wi", "Za", "Ch", "Th", "Sh", "Ph", "Tr", "Gr", "Bl", "Kl"
    ]
    
    FIRST_NAME_SUFFIXES = [
        "ex", "an", "en", "on", "in", "al", "ar", "er", "or", "yn",
        "is", "us", "os", "as", "ed", "id", "od", "ad", "el", "ol",
        "am", "em", "im", "om", "um", "ak", "ek", "ik", "ok", "uk"
    ]
    
    # Last name components
    LAST_NAME_PARTS = [
        "Gre", "Bro", "Wat", "Smi", "Joh", "Wil", "Bro", "Dav", "Mil", "And",
        "Rob", "Lew", "Wri", "Lee", "Har", "Mar", "Gon", "Mor", "Cla", "Gra",
        "Sha", "Whi", "Haw", "Kni", "Tho", "Sco", "Blu", "For", "Hal", "Cra"
    ]
    
    LAST_NAME_SUFFIXES = [
        "son", "ton", "den", "man", "lin", "win", "ford", "wood", "hill", "well",
        "field", "stone", "brook", "vale", "ridge", "brook", "lake", "river", "peak", "shore"
    ]
    
    @staticmethod
    def generate_first_name() -> str:
        """Generate a random first name."""
        part = random.choice(NameGenerator.FIRST_NAME_PARTS)
        suffix = random.choice(NameGenerator.FIRST_NAME_SUFFIXES)
        return part + suffix
    
    @staticmethod
    def generate_last_name() -> str:
        """Generate a random last name."""
        part = random.choice(NameGenerator.LAST_NAME_PARTS)
        suffix = random.choice(NameGenerator.LAST_NAME_SUFFIXES)
        return part + suffix
    
    @staticmethod
    def generate_full_name() -> str:
        """Generate a random full name."""
        first = NameGenerator.generate_first_name()
        last = NameGenerator.generate_last_name()
        return f"{first} {last}"
    
    @staticmethod
    def generate_name() -> str:
        """Generate a random name (alias for generate_full_name)."""
        return NameGenerator.generate_full_name()

