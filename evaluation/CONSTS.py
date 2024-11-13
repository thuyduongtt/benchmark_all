MAX_HOP = 3

METRICS = ['exact_match', 'substring', 'similarity']

# even one is another's substring, they don't have the same meaning
SUBSTRING_EXCEPTIONS = [
    'male___female'
]