from CONSTS import METRICS


class Score:
    exact_match: int = 0
    substring: int = 0
    similarity: float = 0.0

    def __getitem__(self, item):
        if item == 'exact_match':
            return self.exact_match
        if item == 'substring':
            return self.substring
        if item == 'similarity':
            return self.similarity

    def __setitem__(self, key, value):
        if key == 'exact_match':
            self.exact_match = value
        if key == 'substring':
            self.substring = value
        if key == 'similarity':
            self.similarity = value

    def __str__(self):
        return f'exact_match: {self.exact_match}; substring: {self.substring}; similarity: {self.similarity:.4f};'

    def to_list(self):
        return [self[k] for k in METRICS]


class ScoreList:
    exact_match: [int]
    substring: [int]
    similarity: [float]

    def __init__(self):
        self.exact_match = []
        self.substring = []
        self.similarity = []

    def __getitem__(self, item):
        if item == 'exact_match':
            return self.exact_match
        if item == 'substring':
            return self.substring
        if item == 'similarity':
            return self.similarity

    def __setitem__(self, key, value):
        if key == 'exact_match':
            self.exact_match = value
        if key == 'substring':
            self.substring = value
        if key == 'similarity':
            self.similarity = value

    def __str__(self):
        return f'exact_match: {len(self.exact_match)}; substring: {len(self.substring)}; similarity: {len(self.similarity)};'

    def to_list(self):
        return [self[k] for k in METRICS]


if __name__ == '__main__':
    s1 = ScoreList()
    s2 = ScoreList()
    print(s1)
    print(s2)

    s1.substring.append(10)
    s2.similarity.append(1)
    s2.similarity.append(2)
    print(s1)
    print(s2)